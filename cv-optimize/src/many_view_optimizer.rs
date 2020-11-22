use argmin::{
    core::{ArgminOp, Error},
    solver::neldermead::NelderMead,
};
use average::Mean;
use cv_core::nalgebra::{
    dimension::{Dynamic, U1, U4, U6},
    DMatrix, DVector, Matrix3, VecStorage, Vector4, Vector6,
};
use cv_core::{Bearing, Pose, Projective, TriangulatorObservations, WorldPoint, WorldToCamera};
use levenberg_marquardt::LeastSquaresProblem;
use ndarray::{s, Array2};

pub fn many_view_nelder_mead(poses: Vec<WorldToCamera>) -> NelderMead<Array2<f64>, f64> {
    let num_poses = poses.len();
    let se3s: Vec<Vector6<f64>> = poses.iter().map(|p| p.se3()).collect();
    let original = Array2::from_shape_vec(
        (num_poses, 6),
        se3s.iter().flat_map(|se3| se3.iter().copied()).collect(),
    )
    .expect("failed to convert poses into Array2");
    let translation_scale: Mean = original
        .outer_iter()
        .map(|pose| {
            pose.slice(s![0..3])
                .iter()
                .map(|n| n.powi(2))
                .sum::<f64>()
                .sqrt()
                * 0.5
        })
        .collect();
    let mut variants = vec![original; num_poses * 6 + 1];
    #[allow(clippy::needless_range_loop)]
    for i in 0..num_poses * 6 {
        let pose = i / 6;
        let subi = i % 6;
        if subi < 3 {
            // Translation simplex must be relative to existing translation.
            variants[i][(pose, subi)] += translation_scale.mean() * 0.01;
        } else {
            // Rotation simplex must be kept within a small rotation (2 pi would be a complete revolution).
            variants[i][(pose, subi)] += std::f64::consts::PI * 0.001;
        }
    }
    NelderMead::new().with_initial_params(variants)
}

#[derive(Clone)]
pub struct ManyViewConstraint<B, T> {
    loss_cutoff: f64,
    // Stored as a list of landmarks, each of which contains a list of observations in (view, bearing) format.
    landmarks: Vec<Vec<(usize, B)>>,
    triangulator: T,
}

impl<B, T> ManyViewConstraint<B, T>
where
    B: Bearing + Clone,
    T: TriangulatorObservations,
{
    /// Creates a ManyViewConstraint.
    ///
    /// Note that `landmarks` is an iterator over each landmark. Each landmark is an iterator over each
    /// pose presenting the occurence of that landmark in the pose's view. If the landmark doesn't appear in the
    /// view of the pose, the iterator should return `None` for that observance, and Some with the bearing otherwise.
    ///
    /// A landmark is often called a "track" by other MVG software, but the term landmark is preferred to avoid
    /// ambiguity between "camera tracking" and "a track".
    pub fn new<L, O>(landmarks: L, triangulator: T) -> Self
    where
        L: Iterator<Item = O> + Clone,
        O: Iterator<Item = Option<B>>,
        T: TriangulatorObservations,
    {
        let landmarks = landmarks
            .map(|observances| {
                observances
                    .enumerate()
                    .filter_map(|(view, bearing)| bearing.map(|bearing| (view, bearing)))
                    .collect()
            })
            .collect();
        Self {
            loss_cutoff: 0.01,
            landmarks,
            triangulator,
        }
    }

    pub fn loss_cutoff(self, loss_cutoff: f64) -> Self {
        Self {
            loss_cutoff,
            ..self
        }
    }

    pub fn residuals<'a>(
        &'a self,
        poses: impl Iterator<Item = WorldToCamera> + Clone + 'a,
    ) -> impl Iterator<Item = f64> + 'a {
        let loss = move |n: f64| {
            if n > self.loss_cutoff {
                self.loss_cutoff
            } else {
                n
            }
        };
        let poses_res = poses.clone();
        self.landmarks.iter().flat_map(move |observations| {
            if let Some(world_point) =
                self.triangulator
                    .triangulate_observations(observations.iter().map(|(view, bearing)| {
                        (
                        poses.clone().nth(*view).expect(
                            "unexpected pose requested in landmark passed to ManyViewConstraint",
                        ),
                        bearing.clone(),
                    )
                    }))
            {
                let poses_res = poses_res.clone();
                itertools::Either::Left(
                    observations
                        .iter()
                        .map(move |(view, bearing)| {
                            (
                        poses_res.clone().nth(*view).expect(
                            "unexpected pose requested in landmark passed to ManyViewConstraint",
                        ),
                        bearing.clone(),
                    )
                        })
                        .map(move |(pose, lm)| {
                            let camera_point = pose.transform(world_point);
                            loss(1.0 - lm.bearing().dot(&camera_point.bearing()))
                        }),
                )
            } else {
                itertools::Either::Right(
                    std::iter::repeat(self.loss_cutoff).take(observations.len()),
                )
            }
        })
    }
}

impl<B, T> ArgminOp for ManyViewConstraint<B, T>
where
    B: Bearing + Clone,
    T: TriangulatorObservations,
{
    type Param = Array2<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let poses: Vec<WorldToCamera> = p
            .outer_iter()
            .map(|pose_arr| {
                Pose::from_se3(Vector6::from_row_slice(
                    pose_arr.as_slice().expect("param was not contiguous array"),
                ))
            })
            .collect();
        let mean: Mean = self.residuals(poses.iter().copied()).collect();
        Ok(mean.mean())
    }
}

#[derive(Clone)]
pub struct ManyViewOptimizer<B> {
    pub poses: Vec<WorldToCamera>,
    pub points: Vec<Option<WorldPoint>>,
    landmarks: Vec<Vec<Option<B>>>,
    loss_cutoff: f64,
}

impl<B> ManyViewOptimizer<B>
where
    B: Bearing,
{
    /// Creates a ManyViewOptimizer.
    ///
    /// Note that `landmarks` is an iterator over each landmark. Each landmark is an iterator over each
    /// pose presenting the occurence of that landmark in the pose's view. If the landmark doesn't appear in the
    /// view of the pose, the iterator should return `None` for that observance, and Some with the bearing otherwise.
    ///
    /// A landmark is often called a "track" by other MVG software, but the term landmark is preferred to avoid
    /// ambiguity between "camera tracking" and "a track".
    pub fn new<L, O, T>(poses: Vec<WorldToCamera>, landmarks: L, triangulator: T) -> Self
    where
        L: Iterator<Item = O> + Clone,
        O: Iterator<Item = Option<B>>,
        T: TriangulatorObservations,
    {
        let points = landmarks
            .clone()
            .map(|observances| {
                triangulator.triangulate_observations(
                    poses
                        .iter()
                        .copied()
                        .zip(observances)
                        .filter_map(|(pose, observance)| {
                            // Create a tuple of the pose with its corresponding observance.
                            observance.map(move |observance| (pose, observance))
                        }),
                )
            })
            .collect();
        let landmarks = landmarks.map(|observances| observances.collect()).collect();
        Self {
            poses,
            points,
            landmarks,
            loss_cutoff: 0.01,
        }
    }

    pub fn loss_cutoff(self, loss_cutoff: f64) -> Self {
        Self {
            loss_cutoff,
            ..self
        }
    }
}

impl<B> LeastSquaresProblem<f64, Dynamic, Dynamic> for ManyViewOptimizer<B>
where
    B: Bearing + Clone,
{
    /// Storage type used for the residuals. Use `nalgebra::storage::Owned<F, M>`
    /// if you want to use `VectorN` or `MatrixMN`.
    type ResidualStorage = VecStorage<f64, Dynamic, U1>;
    type JacobianStorage = VecStorage<f64, Dynamic, Dynamic>;
    type ParameterStorage = VecStorage<f64, Dynamic, U1>;

    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, params: &DVector<f64>) {
        let poses_len = self.poses.len() * 6;
        for (ix, pose) in self.poses.iter_mut().enumerate() {
            *pose = Pose::from_se3(params.fixed_rows::<U6>(6 * ix).into_owned());
        }
        for (ix, point) in self.points.iter_mut().enumerate() {
            if let Some(p) = point {
                *p = WorldPoint(params.fixed_rows::<U4>(poses_len + 4 * ix).into_owned());
            }
        }
    }

    /// Get the stored parameters `$\vec{x}$`.
    fn params(&self) -> DVector<f64> {
        let pose_len = 6 * self.poses.len();
        let point_len = self.points.len() * 4;
        let zeros = Vector4::zeros();
        let se3s: Vec<Vector6<f64>> = self.poses.iter().map(|p| p.se3()).collect();
        DVector::from_iterator(
            pose_len + point_len,
            se3s.iter().flat_map(|se3| se3.iter().copied()).chain(
                self.points
                    .iter()
                    .flat_map(|p| p.as_ref().map(|p| &p.0).unwrap_or(&zeros).iter().copied()),
            ),
        )
    }

    /// Compute the residual vector.
    fn residuals(&self) -> Option<DVector<f64>> {
        let loss = |n: f64| {
            if n > self.loss_cutoff {
                self.loss_cutoff
            } else {
                n
            }
        };
        Some(DVector::from_iterator(
            self.points.len() * self.poses.len(),
            self.points
                .iter()
                .zip(self.landmarks.iter())
                .flat_map(|(&pw, lms)| {
                    self.poses.iter().zip(lms.iter()).map(move |(pose, lm)| {
                        // TODO: Once try blocks get added, this should be replaced with a try block.
                        let res = || -> Option<f64> {
                            let pc = pose.transform(pw?);
                            Some(loss(1.0 - lm.as_ref()?.bearing().dot(&pc.bearing())))
                        };
                        res().unwrap_or(0.0)
                    })
                }),
        ))
    }

    /// Compute the Jacobian of the residual vector.
    fn jacobian(&self) -> Option<DMatrix<f64>> {
        let pose_len = self.poses.len() * 6;
        let point_len = self.points.len() * 4;
        let mut mat = DMatrix::zeros(self.points.len() * self.poses.len(), pose_len + point_len);
        for (ix, wp, lm, pose) in self
            .points
            .iter()
            .zip(self.landmarks.iter())
            .flat_map(|(&point, lms)| {
                lms.iter()
                    .zip(&self.poses)
                    .map(move |(lm, &pose)| (point, lm.clone(), pose))
            })
            .enumerate()
            .filter_map(|(ix, (point, lm, pose))| Some((ix, point?, lm?, pose)))
        {
            // Get the row corresponding to this index.
            let mut row = mat.row_mut(ix);
            // Transform the point into the camera space and retrieve the jacobians.
            let (cp, jacobian_cp_wp, jacobian_cp_pose) = pose.transform_jacobians(wp);
            // Get the cp bearing and norm.
            let cp_bearing = cp.bearing().into_inner();
            let cp_norm = cp.bearing_unnormalized().norm();
            // The jacobian relating the residual to the normalized form of cp (cpn).
            let jacobian_res_cpn = -lm.bearing().into_inner().transpose();
            // The jacobian relating cpn to cp.
            let jacobian_cpn_cp =
                (Matrix3::identity() - cp_bearing * cp_bearing.transpose()) / cp_norm;

            // The jacobian relating residual to cp (0.0 appended to convert to homogeneous coordinates).
            let jacobian_res_cp = (jacobian_res_cpn * jacobian_cpn_cp)
                .transpose()
                .push(0.0)
                .transpose();

            // Compute the jacobians that will go into the row relating this residual to the pose and the point.
            let jacobian_res_pose = jacobian_res_cp * jacobian_cp_pose;
            let jacobian_res_wp = jacobian_res_cp * jacobian_cp_wp;

            let pose_ix = ix % self.poses.len();
            let point_ix = ix / self.poses.len();
            row.fixed_columns_mut::<U6>(6 * pose_ix)
                .copy_from(&jacobian_res_pose);
            row.fixed_columns_mut::<U4>(pose_len + 4 * point_ix)
                .copy_from(&jacobian_res_wp);
        }
        Some(mat)
    }
}
