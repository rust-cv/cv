mod single_view_optimizer;
mod three_view_optimizer;

pub use single_view_optimizer::*;
pub use three_view_optimizer::*;

use cv_core::nalgebra::{IsometryMatrix3, Rotation3, UnitVector3, Vector3};
use std::ops::{Add, AddAssign};

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

impl AddAssign for Se3TangentSpace {
    fn add_assign(&mut self, rhs: Self) {
        self.translation += rhs.translation;
        self.rotation += rhs.rotation;
    }
}

// `a` must be transformed into the reference frame of the camera being optimized.
// Translation must come from the isometry of the pose from the original reference frame
// of `a` into the reference frame of the camera being optimized.
fn epipolar_gradient(
    translation: Vector3<f64>,
    a: UnitVector3<f64>,
    b: UnitVector3<f64>,
) -> Se3TangentSpace {
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
    // if cross_a_norm < cross_b_norm {
    // Algorithm 12.
    // This effectively computes the sine of the angle between the plane formed between b
    // and translation and the bearing formed by a. It then multiplies this by the normal vector
    // of the plane (nb) to get the normal corrective factor that is applied to a.
    let new_a = UnitVector3::new_normalize(a.into_inner() - (a.dot(&nb) * nb));

    // a can be rotated towards its new bearing.
    Se3TangentSpace {
        translation: -b.cross(&a) * (a.cross(&translation).dot(&b)),
        rotation: a.cross(&new_a),
    }
    // } else {
    //     Se3TangentSpace::identity()
    // }
}
