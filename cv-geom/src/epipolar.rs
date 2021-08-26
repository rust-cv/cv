use cv_core::{
    nalgebra::{UnitVector3, Vector3},
    Se3TangentSpace,
};

// Produces a gradient on the translation and rotation that attempts to fold the two epipolar planes
// into the same plane.
//
// The tangent space acts on the pose that transforms `a` into the same reference frame as `b`.
// The `translation` is the translation extracted from that pose.
#[inline(always)]
pub fn relative_pose_gradient(
    translation: Vector3<f64>,
    a: UnitVector3<f64>,
    b: UnitVector3<f64>,
) -> Se3TangentSpace {
    let nb = b.cross(&translation).normalize();
    let rotation = a.dot(&nb) * nb.cross(&a).normalize();
    Se3TangentSpace::new(a.cross(&b) * (a.cross(&translation).dot(&b)), rotation)
}

/// Produces a gradient that translates a point towards the bearing projecting from the camera.
///
/// The point always exists at the origin. The `translation` describes the position of the optical center
/// of the camera (ray start) relative to the point. The bearing must be transformed to be in the reference frame
/// as the point. It is not relevant which reference frame the point is in, but is typically in world space.
#[inline(always)]
pub fn point_gradient(translation: Vector3<f64>, b: UnitVector3<f64>) -> Vector3<f64> {
    // Reason this projection is reversed is because the translation is reversed from normal,
    // as it comes from the pose directly. Due to this, all the translations are negated.
    // After simplifying, you will find that it simply reverses the order of substraction.
    translation - (translation).dot(&b) * b.into_inner()
}

// Produces a gradient on the translation and rotation that attempts to adjust the pose to make the
// point line up with the bearing. The translation in this case is the translation from the
// optical center of the camera to the point, while `b` is the bearing which matches to the point.
// The translation is in the reference frame of the camera itself.
#[inline(always)]
pub fn world_pose_gradient(translation: Vector3<f64>, b: UnitVector3<f64>) -> Se3TangentSpace {
    let projected_point = (translation).dot(&b) * b.into_inner();
    let translation_gradient = projected_point - translation;
    // let rotation_gradient =
    //     translation_gradient.norm() / projected_point.norm() * translation.cross(&b).normalize();
    Se3TangentSpace::new(translation_gradient, Vector3::zeros())
}

// Produces the absolute value of the sine of the angle between the two epipolar planes.
#[inline(always)]
pub fn loss(translation: Vector3<f64>, a: UnitVector3<f64>, b: UnitVector3<f64>) -> f64 {
    let nb = b.cross(&translation).normalize();
    let corrected_a = (a.into_inner() - (a.dot(&nb) * nb)).normalize();
    let residual = 1.0 - corrected_a.dot(&a);
    // Check chierality as well.
    if residual.is_nan() || corrected_a.dot(&b).is_sign_negative() {
        2.0
    } else {
        residual
    }
}

// // Produces the absolute value of the sine of the angle between the two epipolar planes.
// #[inline(always)]
// pub fn loss(translation: Vector3<f64>, a: UnitVector3<f64>, b: UnitVector3<f64>) -> f64 {
//     let normalized_translation = translation.normalize();
//     // Correct a and b to intersect at the point which minimizes L1 distance as per
//     // "Closed-Form Optimal Two-View Triangulation Based on Angular Errors" algorithm
//     // 12 and 13. The L1 distance minimized is the two angles between the two
//     // epipolar planes and the two bearings.

//     // Compute the cross product of `a` and the unit translation. The magnitude increases as
//     // they become more perpendicular. The unit vector `na` describes the normal of the plane
//     // formed by `a` and the translation.
//     let cross_a = a.cross(&normalized_translation);
//     let cross_a_norm = cross_a.norm();
//     let na = cross_a / cross_a_norm;
//     // Compute the cross product of `b` and the unit translation. The magnitude increases as
//     // they become more perpendicular. The unit vector `nb` describes the normal of the plane
//     // formed by `b` and the translation.
//     let cross_b = b.cross(&normalized_translation);
//     let cross_b_norm = cross_b.norm();
//     let nb = cross_b / cross_b_norm;

//     let res = if cross_a_norm < cross_b_norm {
//         // Algorithm 12.
//         let new_a = UnitVector3::new_normalize(a.into_inner() - (a.dot(&nb) * nb));
//         // Take the cosine distance between the corrected and original bearing.
//         1.0 - new_a.dot(&a)
//     } else {
//         // Algorithm 13.
//         let new_b = UnitVector3::new_normalize(b.into_inner() - (b.dot(&na) * na));
//         // Take the cosine distance between the corrected and original bearing.
//         1.0 - new_b.dot(&b)
//     };

//     if res.is_nan() {
//         2.0
//     } else {
//         res
//     }
// }
