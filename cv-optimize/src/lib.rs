mod single_view_optimizer;
mod three_view_optimizer;

pub use single_view_optimizer::*;
pub use three_view_optimizer::*;

use cv_core::nalgebra::{IsometryMatrix3, Point3, Rotation3, UnitVector3, Vector3};
use std::ops::Add;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
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

// `a` must be transformed into the reference frame of the camera being optimized.
// Translation must come from the isometry of the pose from the original reference frame
// of `a` into the reference frame of the camera being optimized.
fn epipolar_rotation_gradient(
    translation: Vector3<f64>,
    a: UnitVector3<f64>,
    b: UnitVector3<f64>,
) -> Vector3<f64> {
    let normalized_translation = translation.normalize();
    // Correct a and b to intersect at the point which minimizes L1 distance as per
    // "Closed-Form Optimal Two-View Triangulation Based on Angular Errors" algorithm
    // 12 and 13.
    let cross_a = a.cross(&normalized_translation);
    let cross_a_norm = cross_a.norm();
    let cross_b = b.cross(&normalized_translation);
    let cross_b_norm = cross_b.norm();
    let nb = cross_b / cross_b_norm;
    // Shadow the old a and b, as they have been corrected.
    if cross_a_norm < cross_b_norm {
        // Algorithm 12.
        // This effectively computes the sine of the angle between the plane formed between b
        // and translation and the bearing formed by a. It then multiplies this by the normal vector
        // of the plane (nb) to get the normal corrective factor that is applied to a.
        let new_a = UnitVector3::new_normalize(a.into_inner() - (a.dot(&nb) * nb));

        // a can be rotated towards its new bearing.
        a.cross(&new_a)
    } else {
        Vector3::zeros()
    }
}

fn point_translation_gradient(point: Point3<f64>, bearing: UnitVector3<f64>) -> Vector3<f64> {
    let bearing = bearing.into_inner();
    // Find the distance on the observation bearing that the point projects to.
    let projection_distance = point.coords.dot(&bearing);
    // To compute the translation of the camera, we simply look at the translation needed to
    // transform the point itself into the projection of the point onto the bearing.
    // This is counter to the direction we want to move the camera, because the translation is
    // of the world in respect to the camera rather than the camera in respect to the world.
    projection_distance * bearing - point.coords
}
