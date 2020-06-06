use cv_core::nalgebra::{
    dimension::{Dynamic, U1, U6},
    storage::Owned,
    DVector, MatrixMN, VecStorage, Vector3, Vector6,
};
use cv_core::{Bearing, FeatureMatch, Pose, RelativeCameraPose, Skew3, TriangulatorRelative};
use levenberg_marquardt::{differentiate_numerically, LeastSquaresProblem};

#[derive(Clone)]
pub struct TwoViewOptimizer<I, T> {
    matches: I,
    pub pose: RelativeCameraPose,
    triangulator: T,
}

impl<I, P, T> TwoViewOptimizer<I, T>
where
    I: Iterator<Item = FeatureMatch<P>> + Clone,
    P: Bearing,
    T: TriangulatorRelative,
{
    pub fn new(matches: I, pose: RelativeCameraPose, triangulator: T) -> Self {
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
        self.pose.translation.vector = x.xyz();
        let x = x.as_slice();
        self.pose.rotation = Skew3(Vector3::new(x[3], x[4], x[5])).into();
    }

    /// Get the stored parameters `$\vec{x}$`.
    fn params(&self) -> Vector6<f64> {
        let skew: Skew3 = self.pose.rotation.into();
        if let [x, y, z] = *skew.as_slice() {
            self.pose.translation.vector.push(x).push(y).push(z)
        } else {
            unreachable!()
        }
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
                        let a_hat = point.coords.normalize();
                        let b_hat = self.pose.transform(point).coords.normalize();

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
