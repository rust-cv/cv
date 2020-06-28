use cv_core::nalgebra::{
    dimension::{Dynamic, U1, U6},
    storage::Owned,
    DVector, Matrix3, MatrixMN, VecStorage, Vector6,
};
use cv_core::{Bearing, CameraToCamera, FeatureMatch, Pose, Projective, TriangulatorRelative};
use levenberg_marquardt::LeastSquaresProblem;

#[derive(Clone)]
pub struct TwoViewOptimizer<I, T> {
    pub matches: I,
    pub pose: CameraToCamera,
    pub triangulator: T,
}

impl<I, P, T> TwoViewOptimizer<I, T>
where
    I: Iterator<Item = FeatureMatch<P>> + Clone,
    P: Bearing,
    T: TriangulatorRelative,
{
    pub fn new(matches: I, pose: CameraToCamera, triangulator: T) -> Self {
        Self {
            matches,
            pose,
            triangulator,
        }
    }
}

impl<I, P, T> LeastSquaresProblem<f64, Dynamic, U6> for TwoViewOptimizer<I, T>
where
    I: Iterator<Item = FeatureMatch<P>> + Clone,
    P: Bearing + Clone,
    T: TriangulatorRelative + Clone,
{
    /// Storage type used for the residuals. Use `nalgebra::storage::Owned<F, M>`
    /// if you want to use `VectorN` or `MatrixMN`.
    type ResidualStorage = VecStorage<f64, Dynamic, U1>;
    type JacobianStorage = VecStorage<f64, Dynamic, U6>;
    type ParameterStorage = Owned<f64, U6>;

    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, x: &Vector6<f64>) {
        self.pose = Pose::from_se3(Vector6::new(x[0], x[1], x[2], x[3], x[4], x[5]));
    }

    /// Get the stored parameters `$\vec{x}$`.
    fn params(&self) -> Vector6<f64> {
        self.pose.se3()
    }

    /// Compute the residual vector.
    fn residuals(&self) -> Option<DVector<f64>> {
        Some(DVector::from_iterator(
            self.matches.clone().count(),
            self.matches.clone().map(|FeatureMatch(a, b)| {
                self.triangulator
                    .triangulate_relative(self.pose, a, b.clone())
                    .map(move |pa| {
                        let pb = self.pose.transform(pa);
                        let sim_b = pb.bearing().dot(&b.bearing());
                        1.0 - sim_b
                    })
                    .unwrap_or(0.0)
            }),
        ))
    }

    /// Compute the Jacobian of the residual vector.
    fn jacobian(&self) -> Option<MatrixMN<f64, Dynamic, U6>> {
        let mut mat: MatrixMN<f64, Dynamic, U6> =
            MatrixMN::zeros_generic(Dynamic::new(self.matches.clone().count()), U6);
        for (ix, ap, FeatureMatch(_, b)) in
            self.matches
                .clone()
                .enumerate()
                .filter_map(|(ix, FeatureMatch(a, b))| {
                    let ap =
                        self.triangulator
                            .triangulate_relative(self.pose, a.clone(), b.clone())?;
                    Some((ix, ap, FeatureMatch(a, b)))
                })
        {
            // Get the transformed point and the jacobians.
            let (bp, jacobian_bp_pose) = self.pose.transform_jacobian_self(ap);

            // Get the bearings.
            let bp_bearing = bp.bearing().into_inner();
            let bp_xyz_norm = bp.bearing_unnormalized().norm();

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

            // Assign the b_res jacobians for both the pose and the point.
            let mut sim_b_row = mat.row_mut(ix);
            sim_b_row
                .fixed_columns_mut::<U6>(0)
                .copy_from(&jacobian_bres_pose);
        }
        Some(mat)
    }
}
