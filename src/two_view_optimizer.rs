use cv_core::nalgebra::{
    dimension::{Dynamic, U1, U6},
    storage::Owned,
    DVector, MatrixMN, VecStorage, Vector6,
};
use cv_core::{Bearing, CameraToCamera, FeatureMatch, Pose, Projective, TriangulatorRelative};
use levenberg_marquardt::{differentiate_numerically, LeastSquaresProblem};

#[derive(Clone)]
pub struct TwoViewOptimizer<I, T> {
    matches: I,
    pub pose: CameraToCamera,
    triangulator: T,
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
    P: Bearing,
    T: TriangulatorRelative + Clone,
{
    /// Storage type used for the residuals. Use `nalgebra::storage::Owned<F, M>`
    /// if you want to use `VectorN` or `MatrixMN`.
    type ResidualStorage = VecStorage<f64, Dynamic, U1>;
    type JacobianStorage = VecStorage<f64, Dynamic, U6>;
    type ParameterStorage = Owned<f64, U6>;

    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, x: &Vector6<f64>) {
        self.pose = Pose::from_se3(*x);
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
                let a = a.bearing();
                let b = b.bearing();
                self.triangulator
                    .triangulate_relative(self.pose, a, b)
                    .map(|point| {
                        let a_hat = point.bearing();
                        let b_hat = self.pose.transform(point).bearing();

                        1.0 - a.dot(&a_hat) + 1.0 - b.dot(&b_hat)
                    })
                    .unwrap_or(0.0)
            }),
        ))
    }

    /// Compute the Jacobian of the residual vector.
    fn jacobian(&self) -> Option<MatrixMN<f64, Dynamic, U6>> {
        let mut clone = self.clone();
        differentiate_numerically(&mut clone)
    }
}
