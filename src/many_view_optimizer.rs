use alloc::vec::Vec;
use core::iter::once;
use cv_core::nalgebra::{
    dimension::{Dynamic, U1, U3, U6},
    DVector, MatrixMN, Point3, VecStorage, Vector3,
};
use cv_core::{Bearing, Pose, Skew3, TriangulatorObservances, WorldPoint, WorldToCamera};
use levenberg_marquardt::{differentiate_numerically, LeastSquaresProblem};

#[derive(Clone)]
pub struct ManyViewOptimizer<B> {
    pub poses: Vec<WorldToCamera>,
    pub points: Vec<Option<WorldPoint>>,
    landmarks: Vec<Vec<Option<B>>>,
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
        T: TriangulatorObservances,
    {
        let poses: Vec<WorldToCamera> = poses.into();
        let points = landmarks
            .clone()
            .map(|observances| {
                triangulator.triangulate_observances(
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
        }
    }
}

impl<B> LeastSquaresProblem<f64, Dynamic, Dynamic> for ManyViewOptimizer<B>
where
    B: Bearing,
{
    /// Storage type used for the residuals. Use `nalgebra::storage::Owned<F, M>`
    /// if you want to use `VectorN` or `MatrixMN`.
    type ResidualStorage = VecStorage<f64, Dynamic, U1>;
    type JacobianStorage = VecStorage<f64, Dynamic, Dynamic>;
    type ParameterStorage = VecStorage<f64, Dynamic, U1>;

    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, params: &DVector<f64>) {
        // Assert that the number of poses implied by the vector is correct.
        assert_eq!(
            self.poses.len() * 6 + self.points.len() * 3,
            params.len(),
            "number of internal values and length of params are not consistent"
        );
        // Assign the poses
        for (pose, v) in self
            .poses
            .iter_mut()
            .enumerate()
            .map(|(ix, pose)| (pose, params.fixed_rows::<U6>(6 * ix)))
        {
            pose.0.translation.vector = v.xyz();
            pose.0.rotation = Skew3(Vector3::new(v[3], v[4], v[5])).into();
        }
        // Assign the points
        for (point, v) in self
            .points
            .iter_mut()
            .enumerate()
            .filter_map(|(ix, point)| {
                point
                    .as_mut()
                    .map(|p| (p, params.fixed_rows::<U3>(self.poses.len() * 6 + 3 * ix)))
            })
        {
            point.0.coords.copy_from(&v);
        }
    }

    /// Get the stored parameters `$\vec{x}$`.
    fn params(&self) -> DVector<f64> {
        let pose_iter = |pose: &WorldPose| {
            let skew: Skew3 = pose.rotation.into();
            let trans = pose.translation.vector;
            once(trans.x)
                .chain(once(trans.y))
                .chain(once(trans.z))
                .chain(once(skew.x))
                .chain(once(skew.y))
                .chain(once(skew.z))
        };
        let point_iter = |p: Point3<f64>| once(p.x).chain(once(p.y)).chain(once(p.z));
        DVector::from_iterator(
            self.poses.len() * 6,
            self.poses.iter().flat_map(pose_iter).chain(
                self.points
                    .iter()
                    .flat_map(|p| point_iter(p.map(|w| w.0).unwrap_or(Point3::origin()))),
            ),
        )
    }

    /// Compute the residual vector.
    fn residuals(&self) -> Option<DVector<f64>> {
        Some(DVector::from_iterator(
            self.landmarks.clone().count(),
            self.landmarks.clone().map(|observances| {
                self.triangulator
                    .triangulate_observances(
                        self.poses
                            .iter()
                            .copied()
                            .zip(observances.clone())
                            .filter_map(|(pose, observance)| {
                                // Create a tuple of the pose with its corresponding observance.
                                observance.map(move |observance| (CameraPose(pose.0), observance))
                            }),
                    )
                    .map(|point| {
                        // Sum the cosine distance of each bearing to its projected bearing.
                        observances
                            .clone()
                            .zip(&self.poses)
                            .filter_map(|(observance, pose)| {
                                observance.map(|observance| {
                                    1.0 - pose
                                        .transform(point)
                                        .coords
                                        .normalize()
                                        .dot(&observance.bearing())
                                })
                            })
                            .sum::<f64>()
                    })
                    .unwrap_or(0.0)
            }),
        ))
    }

    /// Compute the Jacobian of the residual vector.
    fn jacobian(&self) -> Option<MatrixMN<f64, Dynamic, Dynamic>> {
        let mut clone = self.clone();
        differentiate_numerically(&mut clone)
    }
}
