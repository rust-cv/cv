mod many_view_optimizer;
mod single_view_optimizer;
mod three_view_optimizer;
mod two_view_optimizer;

pub use many_view_optimizer::*;
pub use single_view_optimizer::*;
pub use three_view_optimizer::*;
pub use two_view_optimizer::*;

use cv_core::nalgebra::{IsometryMatrix3, Point3, Rotation3, UnitVector3, Vector3};
use std::ops::Add;

#[derive(Copy, Clone, Debug)]
struct Se3TangentSpace {
    translation: Vector3<f64>,
    rotation: Vector3<f64>,
}

impl Se3TangentSpace {
    fn identity() -> Self {
        Self {
            translation: Vector3::zeros(),
            rotation: Vector3::zeros(),
        }
    }

    /// Gets the isometry that represents this tangent space transformation.
    #[must_use]
    fn isometry(self) -> IsometryMatrix3<f64> {
        let rotation = Rotation3::from_scaled_axis(self.rotation);
        IsometryMatrix3::from_parts((rotation * self.translation).into(), rotation)
    }

    /// Scales both the rotation and the translation.
    #[must_use]
    fn scale(mut self, scale: f64) -> Self {
        self.translation *= scale;
        self.rotation *= scale;
        self
    }
}

impl Add for Se3TangentSpace {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            translation: self.translation + rhs.translation,
            rotation: self.rotation + rhs.rotation,
        }
    }
}

fn observation_gradient(point: Point3<f64>, bearing: UnitVector3<f64>) -> Se3TangentSpace {
    let bearing = bearing.into_inner();
    // Find the distance on the observation bearing that the point projects to.
    let projection_distance = point.coords.dot(&bearing);
    // To compute the translation of the camera, we simply look at the translation needed to
    // transform the point itself into the projection of the point onto the bearing.
    // This is counter to the direction we want to move the camera, because the translation is
    // of the world in respect to the camera rather than the camera in respect to the world.
    let translation = projection_distance * bearing - point.coords;
    // Scale the point so that it would project onto the bearing at unit distance.
    // The reason we do this is so that small distances on this scale are roughly proportional to radians.
    // This is because the first order taylor approximation of `sin(x)` is `x` at `0`.
    // Since we are working with small deltas in the tangent space (SE3), this is an acceptable approximation.
    // TODO: Use loss_cutoff to create a trust region for each sample.
    let scaled = point.coords / projection_distance;
    let delta = scaled - bearing;
    // The delta's norm is now roughly in units of radians, and it points in the direction in the tangent space
    // that we wish to rotate. To compute the so(3) representation of this rotation, we need only take the cross
    // product with the bearing, and this will give us the axis on which we should rotate, with its length
    // roughly proportional to the number of radians.
    let rotation = bearing.cross(&delta);
    Se3TangentSpace {
        translation,
        rotation,
    }
}
