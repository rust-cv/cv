use cv_core::{
    nalgebra::{Point3, UnitVector3, Vector3},
    CameraPoint, Projective, Se3TangentSpace,
};

/// `t` is the translation from B to A.
///
/// Produces a point with A as the origin.
fn two_view_same_space_triangulate_sine_l1(
    t: Vector3<f64>,
    a: UnitVector3<f64>,
    b: UnitVector3<f64>,
) -> Option<Point3<f64>> {
    // Correct a and b to intersect at the point which minimizes L1 distance as per
    // "Closed-Form Optimal Two-View Triangulation Based on Angular Errors" algorithm
    // 12 and 13.
    let cross_a = a.cross(&t);
    let cross_a_norm = cross_a.norm();
    let na = cross_a / cross_a_norm;
    let cross_b = b.cross(&t);
    let cross_b_norm = cross_b.norm();
    let nb = cross_b / cross_b_norm;
    // Shadow the old a and b, as they have been corrected.
    let (a, b) = if cross_a_norm < cross_b_norm {
        // Algorithm 12.
        // This effectively computes the sine of the angle between the plane formed between b
        // and translation and the bearing formed by a. It then multiplies this by the normal vector
        // of the plane (nb) to get the normal corrective factor that is applied to a.
        let new_a = UnitVector3::new_normalize(a.into_inner() - (a.dot(&nb) * nb));
        (new_a, b)
    } else {
        // Algorithm 13.
        // This effectively computes the sine of the angle between the plane formed between a
        // and translation and the bearing formed by b. It then multiplies this by the normal vector
        // of the plane (na) to get the normal corrective factor that is applied to b.
        let new_b = UnitVector3::new_normalize(b.into_inner() - (b.dot(&na) * na));
        (a, new_b)
    };

    let z = a.cross(&b);
    Some(CameraPoint::from_homogeneous(
        a.into_inner().push(z.norm_squared() / z.dot(&t.cross(&b))),
    ))
    .filter(|point| {
        // Ensure the point contains no NaN or infinity.
        point.homogeneous().iter().all(|n| n.is_finite())
    })
    .filter(|point| {
        // Ensure the cheirality constraint.
        point.bearing().dot(&a).is_sign_positive() && point.bearing().dot(&b).is_sign_positive()
    })
    .and_then(|p| p.point())
}

// Computes the rotation gradient where t goes from a to b.
fn two_view_rotation_gradient(
    t: Vector3<f64>,
    a: UnitVector3<f64>,
    b: UnitVector3<f64>,
) -> Vector3<f64> {
    // Compute the cross product of `a` and the unit translation. The magnitude increases as
    // they become more perpendicular. The unit vector `na` describes the normal of the plane
    // formed by `a` and the translation. This is the norm of the epipolar plane formed by A.
    let cross_a = a.cross(&t);
    // Compute the cross product of `b` and the unit translation. The magnitude increases as
    // they become more perpendicular. The unit vector `nb` describes the normal of the plane
    // formed by `b` and the translation. This is the norm of the epipolar plane formed by B.
    let cross_b = b.cross(&t);
    // Take the epipolar plane norm of A and attempt to rotate it to become parallel to B's epipolar plane norm.
    cross_b.normalize().cross(&cross_a.normalize())
}

// Produces a gradient on the translation and rotation.
//
// Must be provided:
// * `c` - the center camera bearing in the center camera reference frame
// * `f` - the first camera bearing in the center camera reference frame
// * `cft` - the translation from the center camera to the first camera in the center camera reference frame
// * `s` - the second camera bearing in the center camera reference frame
// * `cst` - the translation from the center camera to the second camera in the center camera reference frame
//
// Returns the array of `[first_pose_gradient, second_pose_gradient]` in the tangent space of the center camera's
// reference frame.
#[inline(always)]
pub fn three_view_gradients(
    c: UnitVector3<f64>,
    f: UnitVector3<f64>,
    ftoc: Vector3<f64>,
    s: UnitVector3<f64>,
    stoc: Vector3<f64>,
) -> [Se3TangentSpace; 2] {
    // The translation from first to second cameras.
    let stof = stoc - ftoc;

    let rot_cf = two_view_rotation_gradient(ftoc, c, f);
    let rot_cs = two_view_rotation_gradient(stoc, c, s);
    // This is a rotation in the space of camera C that would move the pose of camera S.
    let rot_fs = two_view_rotation_gradient(stof, f, s);
    // The negative rotation is applied to the first pose, it is in the opposite direction.
    let rot_sf = -rot_fs;

    // The rotation gradients above are in the rotational reference frame of camera C, and they
    // will all be returned that way, as we are optimizing the rotations and translations in this
    // reference frame. They describe the rotation needed to turn the second parameter into the first.
    // All of the rotations with the second bearing set to `f` need to go to the first gradient and
    // the rotations with the second bearing set to `s` need to go to the second gradient.
    // One caveat is that `rot_fs` and `rot_sf` are from the same edge, and so they must supply half
    // the gradient, because it is actually the same gradient being applied to two poses to
    // even out the application of gradient. We could have a gradient for the center camera, which
    // would make this a non-issue, but those would be three unecessary degrees of freedom.
    let first_rotation = rot_cf * (2.0 / 3.0) + rot_sf * (1.0 / 3.0);
    let second_rotation = rot_cs * (2.0 / 3.0) + rot_fs * (1.0 / 3.0);
    // let first_rotation = rot_cf;
    // let second_rotation = rot_cs;

    // For the translation gradients, compute the point for each set of 2 and try to translate
    // the third camera towards the intersection point of the other two cameras.
    let trans_f = two_view_same_space_triangulate_sine_l1(-stoc, c, s)
        .map(|p| {
            // Move the point into the translational reference frame of F.
            let p = p - ftoc;
            // Project the point onto the bearing `f`.
            let projection = p.coords.dot(&f) * f.into_inner();
            // The translation gradient is directed to make ftoc move in the right direction.
            p.coords - projection
        })
        .unwrap_or_else(Vector3::zeros);
    let trans_s = two_view_same_space_triangulate_sine_l1(-ftoc, c, f)
        .map(|p| {
            // Move the point into the translational reference frame of S.
            let p = p - stoc;
            // Project the point onto the bearing `s`.
            let projection = p.coords.dot(&s) * s.into_inner();
            // The translation gradient is directed to make stoc move in the right direction.
            p.coords - projection
        })
        .unwrap_or_else(Vector3::zeros);

    let trans_c = two_view_same_space_triangulate_sine_l1(-stof, f, s)
        .map(|p| {
            // Move the point into the translational reference frame of C from F.
            let p = p + ftoc;
            // Project the point onto the bearing `c`.
            let projection = p.coords.dot(&c) * c.into_inner();
            // The translation gradient is directed to make ftoc and stoc move in the right direction.
            projection - p.coords
        })
        .unwrap_or_else(Vector3::zeros);

    // To compute the final translation gradients, we need to do the same thing we did with the rotation
    // gradients. We need to apply `trans_f` and `trans_s` to the proper poses. However, since we do not translate
    // the center pose (leaving it as the origin), we need to apply the translation gradient for `trans_c` to
    // the first and second poses. To do this, we just give half of it to each pose, and we need to basically
    // apply the opposite translation to those poses.
    let first_translation = trans_f * (2.0 / 3.0) + trans_c * (1.0 / 3.0);
    let second_translation = trans_s * (2.0 / 3.0) + trans_c * (1.0 / 3.0);
    // let first_translation = trans_f;
    // let second_translation = trans_s;
    // let first_translation = Vector3::zeros();
    // let second_translation = Vector3::zeros();

    [
        Se3TangentSpace::new(first_translation, first_rotation),
        Se3TangentSpace::new(second_translation, second_rotation),
    ]
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
//
// Returns the L2 tangent space delta.
#[inline(always)]
pub fn world_pose_gradient(translation: Vector3<f64>, b: UnitVector3<f64>) -> Se3TangentSpace {
    let projected_point = translation.dot(&b) * b.into_inner();
    let translation_gradient = projected_point - translation;
    let rotation_gradient = translation.normalize().cross(&b);
    Se3TangentSpace::new(translation_gradient, rotation_gradient)
}

// Produces the absolute value of the sine of the angle between the two epipolar planes.
#[inline(always)]
pub fn loss(translation: Vector3<f64>, a: UnitVector3<f64>, b: UnitVector3<f64>) -> f64 {
    // Correct a and b to intersect at the point which minimizes L1 distance as per
    // "Closed-Form Optimal Two-View Triangulation Based on Angular Errors" algorithm
    // 12 and 13. The L1 distance minimized is the two angles between the two
    // epipolar planes and the two bearings.

    // Compute the cross product of `a` and the unit translation. The magnitude increases as
    // they become more perpendicular. The unit vector `na` describes the normal of the plane
    // formed by `a` and the translation.
    let cross_a = a.cross(&translation);
    let cross_a_norm_squared = cross_a.norm_squared();
    // Compute the cross product of `b` and the unit translation. The magnitude increases as
    // they become more perpendicular. The unit vector `nb` describes the normal of the plane
    // formed by `b` and the translation.
    let cross_b = b.cross(&translation);
    let cross_b_norm_squared = cross_b.norm_squared();

    let residual = if cross_a_norm_squared < cross_b_norm_squared {
        // If `a` is less perpendicular to the translation, we compute the projection length of `a`
        // onto `b`'s epipolar plane normal (how far it is out of the epipolar plane) and then
        // take the absolute value to get sine distance.
        a.dot(&cross_b.scale(cross_b_norm_squared.sqrt().recip()))
            .abs()
    } else {
        // If `b` is less perpendicular to the translation, we compute the projection length of `b`
        // onto `a`'s epipolar plane normal (how far it is out of the epipolar plane) and then
        // take the absolute value to get sine distance.
        b.dot(&cross_a.scale(cross_a_norm_squared.sqrt().recip()))
            .abs()
    };
    // Check chierality as well.
    if residual.is_nan() || a.dot(&b).is_sign_negative() {
        1.0
    } else {
        residual
    }
}
