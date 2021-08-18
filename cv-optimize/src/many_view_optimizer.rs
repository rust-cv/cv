use std::iter::{once, repeat};

use argmin::{
    core::{ArgminOp, Error},
    solver::neldermead::NelderMead,
};
use average::Mean;
use cv_core::{
    nalgebra::{
        dimension::{Dynamic, U1},
        DMatrix, DVector, UnitVector3, VecStorage, Vector6,
    },
    CameraToCamera, Pose, Projective, TriangulatorObservations, WorldToCamera,
};
use itertools::{Either, Itertools};
use levenberg_marquardt::LeastSquaresProblem;

pub fn many_view_nelder_mead(poses: Vec<WorldToCamera>) -> NelderMead<Vec<Vec<f64>>, f64> {
    let num_poses = poses.len();
    let se3s: Vec<Vec<f64>> = poses
        .iter()
        .map(|p| p.se3().iter().copied().collect())
        .collect();
    let translation_scale: Mean = se3s
        .iter()
        .map(|pose| pose[..3].iter().map(|n| n.powi(2)).sum::<f64>().sqrt() * 0.5)
        .collect();
    let mut variants = vec![se3s; num_poses * 6 + 1];
    #[allow(clippy::needless_range_loop)]
    for i in 0..num_poses * 6 {
        let pose = i / 6;
        let subi = i % 6;
        if subi < 3 {
            // Translation simplex must be relative to existing translation.
            variants[i][pose][subi] += translation_scale.mean() * 0.01;
        } else {
            // Rotation simplex must be kept within a small rotation (2 pi would be a complete revolution).
            variants[i][pose][subi] += std::f64::consts::PI * 0.001;
        }
    }
    NelderMead::new().with_initial_params(variants)
}

#[derive(Clone)]
pub struct StructurelessManyViewOptimizer<T> {
    loss_cutoff: f64,
    // Stored as a list of landmarks, each of which contains a list of observations in (view, bearing) format.
    landmarks: Vec<Vec<(usize, UnitVector3<f64>)>>,
    triangulator: T,
}

impl<T> StructurelessManyViewOptimizer<T>
where
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
        O: Iterator<Item = Option<UnitVector3<f64>>>,
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
                    .triangulate_observations(observations.iter().map(|&(view, bearing)| {
                        (
                        poses.clone().nth(view).expect(
                            "unexpected pose requested in landmark passed to ManyViewConstraint",
                        ),
                        bearing,
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
                        bearing,
                    )
                        })
                        .map(move |(pose, bearing)| {
                            let camera_point = pose.transform(world_point);
                            loss(1.0 - bearing.dot(&camera_point.bearing()))
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

impl<T> ArgminOp for StructurelessManyViewOptimizer<T>
where
    T: TriangulatorObservations,
{
    type Param = Vec<Vec<f64>>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let poses: Vec<WorldToCamera> = p
            .iter()
            .map(|pose_arr| Pose::from_se3(Vector6::from_row_slice(pose_arr)))
            .collect();
        let mean: Mean = self.residuals(poses.iter().copied()).collect();
        Ok(mean.mean())
    }
}

#[derive(Clone)]
pub struct ManyViewOptimizer<T> {
    pub poses: Vec<CameraToCamera>,
    landmarks: Vec<UnitVector3<f64>>,
    loss_cutoff: f64,
    triangulator: T,
    residuals: DVector<f64>,
    jacobian: DMatrix<f64>,
}

impl<T> ManyViewOptimizer<T>
where
    T: TriangulatorObservations,
{
    /// Creates a ManyViewOptimizer.
    ///
    /// You must pass at least 2 poses. It is highly recommended to pass at least 3 poses.
    /// If you want 2 poses, use [`TwoViewOptimizer`](crate::TwoViewOptimizer), which is optimized for that case.
    ///
    /// Note that `landmarks` must be formatted such that it contains each landmark's observations contiguously.
    /// There must be an observation for every pose, and each landmark must have exactly that many observations.
    /// For instance, if there are 2 poses (1, 2) and three landmarks (a, b, c), the vector should have this layout:
    ///
    /// ```ignore
    /// [a1, a2, b1, b2, c1, c2]
    /// ```
    ///
    /// Due to this, the number of items in the vector must be a multiple of the number of poses.
    ///
    /// This optimizer produces one less pose than is passed in. The first pose passed in is considered to be the
    /// origin. Every other pose produced is relative to the first pose. There is also no guarantee of scale
    /// consistency, so make sure that you correct for the resulting scale (translation) of the poses.
    pub fn new(
        loss_cutoff: f64,
        mut poses: impl Iterator<Item = WorldToCamera>,
        landmarks: Vec<UnitVector3<f64>>,
        triangulator: T,
    ) -> Self {
        let center = poses
            .next()
            .expect("you must pass at least one pose to ThreeViewOptimizier");

        let poses = poses
            .map(|pose| CameraToCamera(pose.isometry() * center.isometry().inverse()))
            .collect_vec();

        let (residuals, jacobian) =
            compute_residuals_and_jacobian(loss_cutoff, &poses, &landmarks, &triangulator);
        Self {
            poses,
            landmarks,
            loss_cutoff,
            triangulator,
            residuals,
            jacobian,
        }
    }

    pub fn loss_cutoff(self, loss_cutoff: f64) -> Self {
        Self {
            loss_cutoff,
            ..self
        }
    }
}

impl<T> LeastSquaresProblem<f64, Dynamic, Dynamic> for ManyViewOptimizer<T>
where
    T: TriangulatorObservations,
{
    /// Storage type used for the residuals. Use `nalgebra::storage::Owned<F, M>`
    /// if you want to use `OVector` or `OMatrix`.
    type ResidualStorage = VecStorage<f64, Dynamic, U1>;
    type JacobianStorage = VecStorage<f64, Dynamic, Dynamic>;
    type ParameterStorage = VecStorage<f64, Dynamic, U1>;

    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, params: &DVector<f64>) {
        for (ix, pose) in self.poses.iter_mut().enumerate() {
            *pose = Pose::from_se3(params.fixed_rows::<6>(6 * ix).into_owned());
        }
        let (residuals, jacobian) = compute_residuals_and_jacobian(
            self.loss_cutoff,
            &self.poses,
            &self.landmarks,
            &self.triangulator,
        );
        self.residuals = residuals;
        self.jacobian = jacobian;
    }

    /// Get the stored parameters `$\vec{x}$`.
    fn params(&self) -> DVector<f64> {
        let pose_len = 6 * self.poses.len();
        let se3s: Vec<Vector6<f64>> = self.poses.iter().map(|p| p.se3()).collect();
        DVector::from_iterator(pose_len, se3s.iter().flat_map(|se3| se3.iter().copied()))
    }

    /// Compute the residual vector.
    fn residuals(&self) -> Option<DVector<f64>> {
        Some(self.residuals.clone())
    }

    /// Compute the Jacobian of the residual vector.
    fn jacobian(&self) -> Option<DMatrix<f64>> {
        Some(self.jacobian.clone())
    }
}

fn compute_residuals_and_jacobian(
    loss_cutoff: f64,
    poses: &[CameraToCamera],
    landmarks: &[UnitVector3<f64>],
    triangulator: &impl TriangulatorObservations,
) -> (DVector<f64>, DMatrix<f64>) {
    let loss = |n: f64| {
        if n > loss_cutoff {
            loss_cutoff
        } else {
            n
        }
    };
    let residuals = DVector::from_iterator(
        landmarks.len(),
        landmarks
            .chunks_exact(poses.len() + 1)
            .flat_map(|observations| {
                if let Some(point) = triangulator.triangulate_observations_to_camera(
                    observations[0],
                    poses.iter().copied().zip(observations[1..].iter().copied()),
                ) {
                    let first_residual = once(loss(1.0 - point.bearing().dot(&observations[0])));
                    let other_residuals =
                        poses
                            .iter()
                            .zip(observations[1..].iter())
                            .map(move |(pose, bearing)| {
                                loss(1.0 - pose.transform(point).bearing().dot(bearing))
                            });
                    Either::Left(first_residual.chain(other_residuals))
                } else {
                    Either::Right(repeat(loss_cutoff).take(poses.len() + 1))
                }
            }),
    );

    let mut jacobian = DMatrix::zeros(landmarks.len(), poses.len() * 6);
    for (landmark_ix, observations) in landmarks.chunks_exact(poses.len() + 1).enumerate() {
        // If the point is not triangulatable, then there is zero gradient as the loss is fixed, so the initial
        // zeros are correct.
        if let Some(_point) = triangulator.triangulate_observations_to_camera(
            observations[0],
            poses.iter().copied().zip(observations[1..].iter().copied()),
        ) {
            for (observation_ix, _bearing) in observations.iter().enumerate() {
                // Get the row corresponding to this residual (observation).
                let mut _row = jacobian.row_mut(landmark_ix * (poses.len() + 1) + observation_ix);
                // Go through each pose, and we will compute the partial derivatives of that pose in respect to the
                for (_pose_ix, _pose) in poses.iter().enumerate() {
                    // Not finished yet.
                    unimplemented!();
                }
            }
        }
    }

    (residuals, jacobian)
}
