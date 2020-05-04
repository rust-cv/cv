use core::iter::once;
use cv_core::nalgebra::{
    dimension::{Dynamic, U1, U6},
    storage::Owned,
    DVector, IsometryMatrix3, Matrix3, Matrix6x3, MatrixMN, VecStorage, Vector3, Vector6,
};
use cv_core::{Bearing, FeatureWorldMatch, Skew3, WorldPose};
use levenberg_marquardt::LeastSquaresProblem;

#[derive(Clone)]
pub struct WorldPoseFromFeatureWorldMatches<I> {
    matches: I,
    translation: Vector3<f64>,
    rotation: Skew3,
}

impl<I> WorldPoseFromFeatureWorldMatches<I> {
    pub fn new(matches: I, pose: WorldPose) -> Self {
        Self {
            matches,
            translation: pose.translation.vector,
            rotation: pose.rotation.into(),
        }
    }
}

impl<I, P> LeastSquaresProblem<f64, Dynamic, U6> for WorldPoseFromFeatureWorldMatches<I>
where
    I: Iterator<Item = FeatureWorldMatch<P>> + Clone,
    P: Bearing,
{
    /// Storage type used for the residuals. Use `nalgebra::storage::Owned<F, M>`
    /// if you want to use `VectorN` or `MatrixMN`.
    type ResidualStorage = VecStorage<f64, Dynamic, U1>;
    type JacobianStorage = VecStorage<f64, Dynamic, U6>;
    type ParameterStorage = Owned<f64, U6>;

    /// Set the stored parameters `$\vec{x}$`.
    fn set_params(&mut self, x: &Vector6<f64>) {
        self.translation = x.xyz();
        let x = x.as_slice();
        self.rotation = Skew3(Vector3::new(x[3], x[4], x[5]));
    }

    /// Get the stored parameters `$\vec{x}$`.
    fn params(&self) -> Vector6<f64> {
        if let [x, y, z] = *self.rotation.0.as_slice() {
            self.translation.push(x).push(y).push(z)
        } else {
            unreachable!()
        }
    }

    /// Compute the residual vector.
    fn residuals(&self) -> Option<DVector<f64>> {
        let pose = WorldPose(IsometryMatrix3::new(self.translation, self.rotation.0));
        Some(DVector::from_iterator(
            self.matches.clone().count(),
            self.matches
                .clone()
                .map(|FeatureWorldMatch(feature, world)| {
                    // Get the normalized bearing.
                    let feature_bearing = feature.bearing();
                    // Transform the world point to the camera space and then normalize it to create a bearing.
                    let world_bearing = pose.transform(world).coords.normalize();
                    // The cosine distance is our residual metric.
                    1.0 - feature_bearing.dot(&world_bearing)
                }),
        ))
    }

    /// Compute the Jacobian of the residual vector.
    fn jacobian(&self) -> Option<MatrixMN<f64, Dynamic, U6>> {
        let pose = WorldPose(IsometryMatrix3::new(self.translation, self.rotation.0));
        // Number of correspondences (and hence residuals).
        let rows = Dynamic::new(self.matches.clone().count());
        // Create the jacobian.
        Some(MatrixMN::from_iterator_generic(
            rows,
            U6,
            self.matches
                .clone()
                .flat_map(|FeatureWorldMatch(feature, world)| {
                    // Rotated point (intermediate output)
                    let pr = (pose.0.rotation * world.0).coords;

                    // dP/dT (Jacobian of camera point in respect to translation component)
                    let dp_dt = Matrix3::<f64>::identity();

                    // dP/dR
                    let dp_dr = Skew3::jacobian_output_to_self(pr);

                    // dP/dT,Q (Jacobian of 3d camera point in respect to translation and skew)
                    let dp_dtq = Matrix6x3::<f64>::from_rows(&[
                        dp_dt.row(0),
                        dp_dt.row(1),
                        dp_dt.row(2),
                        dp_dr.row(0),
                        dp_dr.row(1),
                        dp_dr.row(2),
                    ]);

                    // Get the normalized bearing.
                    let feature_bearing = feature.bearing();
                    // Transform the world point to the camera space and then normalize it to create a bearing.
                    let world_bearing = pose.transform(world).coords.normalize();
                    // The cosine distance is our residual metric.
                    let components = dp_dtq * -feature_bearing.component_mul(&world_bearing);
                    once(components[0])
                        .chain(once(components[1]))
                        .chain(once(components[2]))
                        .chain(once(components[3]))
                        .chain(once(components[4]))
                        .chain(once(components[5]))
                }),
        ))
    }
}
