use argmin::{
    core::{ArgminOp, Error},
    solver::neldermead::NelderMead,
};
use average::Mean;
use core::iter::once;
use cv_core::nalgebra::{
    dimension::{Dynamic, U1, U4, U6},
    DMatrix, DVector, Matrix3, VecStorage, Vector4, Vector6,
};
use cv_core::{
    Bearing, CameraPoint, CameraToCamera, FeatureMatch, Pose, Projective, TriangulatorRelative,
};
use levenberg_marquardt::LeastSquaresProblem;
use ndarray::{s, Array1};

pub fn two_view_nelder_mead(pose: CameraToCamera) -> NelderMead<Array1<f64>, f64> {
    let original = Array1::from(pose.se3().iter().copied().collect::<Vec<f64>>());
    let translation_scale = original
        .slice(s![0..3])
        .iter()
        .map(|n| n.powi(2))
        .sum::<f64>()
        .sqrt()
        * 0.001;
    let mut variants = vec![original; 7];
    #[allow(clippy::needless_range_loop)]
    for i in 0..6 {
        if i < 3 {
            // Translation simplex must be relative to existing translation.
            variants[i][i] += translation_scale;
        } else {
            // Rotation simplex must be kept within a small rotation (2 pi would be a complete revolution).
            variants[i][i] += std::f64::consts::PI * 0.0001;
        }
    }
    NelderMead::new().with_initial_params(variants)
}

#[derive(Clone)]
pub struct TwoViewConstraint<I, T> {
    loss_cutoff: f64,
    matches: I,
    triangulator: T,
}

impl<I, P, T> TwoViewConstraint<I, T>
where
    I: Iterator<Item = FeatureMatch<P>> + Clone,
    P: Bearing,
    T: TriangulatorRelative,
{
    pub fn new(matches: I, triangulator: T) -> Self {
        Self {
            loss_cutoff: 0.001,
            matches,
            triangulator,
        }
    }

    pub fn loss_cutoff(self, loss_cutoff: f64) -> Self {
        Self {
            loss_cutoff,
            ..self
        }
    }

    pub fn residuals(&self, pose: CameraToCamera) -> impl Iterator<Item = f64> + '_
    where
        P: Clone,
    {
        self.matches.clone().flat_map(move |FeatureMatch(a, b)| {
            if let Some(pa) = self
                .triangulator
                .triangulate_relative(pose, a.clone(), b.clone())
            {
                let pb = pose.transform(pa);
                let sim_a = pa.bearing().dot(&a.bearing());
                let sim_b = pb.bearing().dot(&b.bearing());
                let loss = |n: f64| {
                    if n > self.loss_cutoff {
                        self.loss_cutoff
                    } else {
                        n
                    }
                };
                once(loss(1.0 - sim_a)).chain(once(loss(1.0 - sim_b)))
            } else {
                once(self.loss_cutoff).chain(once(self.loss_cutoff))
            }
        })
    }
}

impl<I, P, T> ArgminOp for TwoViewConstraint<I, T>
where
    I: Iterator<Item = FeatureMatch<P>> + Clone,
    P: Bearing + Clone,
    T: TriangulatorRelative + Clone,
{
    type Param = Array1<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let pose = Pose::from_se3(Vector6::from_row_slice(
            p.as_slice().expect("param was not contiguous array"),
        ));
        let mean: Mean = self.residuals(pose).collect();
        Ok(mean.mean())
    }
}

#[derive(Clone)]
pub struct TwoViewOptimizer<I, T> {
    pub pose: CameraToCamera,
    pub loss_cutoff: f64,
    matches: I,
    points: Vec<Option<CameraPoint>>,
    triangulator: T,
}

impl<I, P, T> TwoViewOptimizer<I, T>
where
    I: Iterator<Item = FeatureMatch<P>> + Clone,
    P: Bearing,
    T: TriangulatorRelative,
{
    pub fn new(matches: I, pose: CameraToCamera, triangulator: T) -> Self {
        let points = matches
            .clone()
            .map(|FeatureMatch(a, b)| triangulator.triangulate_relative(pose, a, b))
            .collect();
        Self {
            pose,
            loss_cutoff: 0.001,
            matches,
            points,
            triangulator,
        }
    }

    pub fn loss_cutoff(self, loss_cutoff: f64) -> Self {
        Self {
            loss_cutoff,
            ..self
        }
    }
}

impl<I, P, T> LeastSquaresProblem<f64, Dynamic, Dynamic> for TwoViewOptimizer<I, T>
where
    I: Iterator<Item = FeatureMatch<P>> + Clone,
    P: Bearing,
    T: TriangulatorRelative + Clone,
{
    /// Storage type used for the residuals. Use `nalgebra::storage::Owned<F, M>`
    /// if you want to use `OVector` or `OMatrix`.
    type ResidualStorage = VecStorage<f64, Dynamic, U1>;
    type JacobianStorage = VecStorage<f64, Dynamic, Dynamic>;
    type ParameterStorage = VecStorage<f64, Dynamic, U1>;

    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, x: &DVector<f64>) {
        self.pose = Pose::from_se3(Vector6::new(x[0], x[1], x[2], x[3], x[4], x[5]));
        for (ix, point) in self.points.iter_mut().enumerate() {
            if let Some(p) = point {
                *p = CameraPoint(x.fixed_rows::<U4>(6 + 4 * ix).into_owned());
            }
        }
    }

    /// Get the stored parameters `$\vec{x}$`.
    fn params(&self) -> DVector<f64> {
        let pose_len = 6;
        let point_len = self.points.len() * 4;
        let zeros = Vector4::zeros();
        DVector::from_iterator(
            pose_len + point_len,
            self.pose.se3().iter().copied().chain(
                self.points
                    .iter()
                    .flat_map(|p| p.as_ref().map(|p| &p.0).unwrap_or(&zeros).iter().copied()),
            ),
        )
    }

    /// Compute the residual vector.
    fn residuals(&self) -> Option<DVector<f64>> {
        Some(DVector::from_iterator(
            self.points.len() * 2,
            self.points
                .iter()
                .zip(self.matches.clone())
                .flat_map(|(pa, FeatureMatch(a, b))| {
                    if let Some(pa) = *pa {
                        let pb = self.pose.transform(pa);
                        let sim_a = pa.bearing().dot(&a.bearing());
                        let sim_b = pb.bearing().dot(&b.bearing());
                        let loss = |n: f64| {
                            if n > self.loss_cutoff {
                                self.loss_cutoff
                            } else {
                                n
                            }
                        };
                        once(loss(1.0 - sim_a)).chain(once(loss(1.0 - sim_b)))
                    } else {
                        once(0.0).chain(once(0.0))
                    }
                }),
        ))
    }

    /// Compute the Jacobian of the residual vector.
    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let pose_len = 6;
        let point_len = self.points.len() * 4;
        let mut mat = DMatrix::zeros(self.points.len() * 2, pose_len + point_len);
        for (ix, ap, FeatureMatch(a, b)) in self
            .points
            .iter()
            .copied()
            .zip(self.matches.clone())
            .enumerate()
            .filter_map(|(ix, (ap, m))| Some((ix, ap?, m)))
        {
            // Get the transformed point and the jacobians.
            let (bp, jacobian_bp_ap, jacobian_bp_pose) = self.pose.transform_jacobians(ap);

            // Get the bearings.
            let ap_bearing = ap.bearing().into_inner();
            let ap_xyz_norm = ap.bearing_unnormalized().norm();
            let bp_bearing = bp.bearing().into_inner();
            let bp_xyz_norm = bp.bearing_unnormalized().norm();

            // The jacobian relating the a residual to the normalized form of ap.
            let jacobian_ares_ap_bearing = -a.bearing().into_inner().transpose();
            // The jacobian relating the normalized form of ap to ap.
            let jacobian_ap_bearing =
                (Matrix3::identity() - ap_bearing * ap_bearing.transpose()) / ap_xyz_norm;

            // Form the jacobian relating the a residual to ap.
            let jacobian_ares_ap = (jacobian_ares_ap_bearing * jacobian_ap_bearing)
                .transpose()
                .push(0.0)
                .transpose();

            // The jacobian relating the b residual to the normalized form of bp.
            let jacobian_bres_bp_bearing = -b.bearing().into_inner().transpose();
            // The jacobian relating the normalized form of bp to bp.
            let jacobian_bp_bearing =
                (Matrix3::identity() - bp_bearing * bp_bearing.transpose()) / bp_xyz_norm;

            // Form the jacobian relating the b residual to bp.
            // A 0 is appended because the homogeneous dimension has no effect (the transpose is needed due to nalgebra prefering column vectors).
            let jacobian_bres_bp = (jacobian_bres_bp_bearing * jacobian_bp_bearing)
                .transpose()
                .push(0.0)
                .transpose();

            // Form the jacobian relating the b residual to the pose.
            let jacobian_bres_pose = jacobian_bres_bp * jacobian_bp_pose;
            // Form the jacobian relating the b residual to ap.
            let jacobian_bres_ap = jacobian_bres_bp * jacobian_bp_ap;
            // Within the b row, the first 6 columns are for the pose.

            // Assign the a_res jacobian for the point only (pose doesn't affect this one).
            let mut sim_a_row = mat.row_mut(ix * 2);
            sim_a_row
                .fixed_columns_mut::<U4>(6 + 4 * ix)
                .copy_from(&jacobian_ares_ap);

            // Assign the b_res jacobians for both the pose and the point.
            let mut sim_b_row = mat.row_mut(ix * 2 + 1);
            sim_b_row
                .fixed_columns_mut::<U6>(0)
                .copy_from(&jacobian_bres_pose);
            sim_b_row
                .fixed_columns_mut::<U4>(6 + 4 * ix)
                .copy_from(&jacobian_bres_ap);
        }
        Some(mat)
    }
}
