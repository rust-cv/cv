use crate::AdaMaxSo3Tangent;
use cv_core::{FeatureWorldMatch, Pose, Projective, Se3TangentSpace, WorldToCamera};
use cv_geom::epipolar;
use float_ord::FloatOrd;
use itertools::izip;

fn landmark_delta(pose: WorldToCamera, landmark: FeatureWorldMatch) -> Option<Se3TangentSpace> {
    let FeatureWorldMatch(bearing, world_point) = landmark;
    let camera_point = pose.transform(world_point);
    Some(epipolar::world_pose_gradient(
        camera_point.point()?.coords,
        bearing,
    ))
}

pub fn single_view_simple_optimize_l1(
    original_pose: WorldToCamera,
    translation_trust_region: f64,
    rotation_trust_region: f64,
    iterations: usize,
    landmarks: &[FeatureWorldMatch],
) -> WorldToCamera {
    if landmarks.is_empty() {
        return original_pose;
    }
    let mut optimizer = AdaMaxSo3Tangent::new(
        original_pose.isometry(),
        translation_trust_region,
        rotation_trust_region,
        1.0,
    );
    let inv_landmark_len = (landmarks.len() as f64).recip();
    let mut translations = vec![0.0; landmarks.len()];
    let mut rotations = vec![0.0; landmarks.len()];
    for iteration in 0..iterations {
        let mut l1 = Se3TangentSpace::identity();
        for (t, r, &landmark) in izip!(
            translations.iter_mut(),
            rotations.iter_mut(),
            landmarks.iter()
        ) {
            if let Some(tangent) = landmark_delta(WorldToCamera(optimizer.pose()), landmark) {
                *t = tangent.translation.norm();
                *r = tangent.rotation.norm();
                l1 += tangent.l1();
            } else {
                *t = 0.0;
                *r = 0.0;
            }
        }
        translations.sort_unstable_by_key(|&f| FloatOrd(f));
        rotations.sort_unstable_by_key(|&f| FloatOrd(f));

        let translation_scale = translations[translations.len() / 2] * inv_landmark_len;
        let rotation_scale = rotations[rotations.len() / 2] * inv_landmark_len;

        let tangent = l1
            .scale(inv_landmark_len)
            .scale_translation(translation_scale)
            .scale_rotation(rotation_scale);

        if optimizer.step(tangent) {
            log::info!(
                "terminating single-view optimization due to stabilizing on iteration {}",
                iteration
            );
            log::info!("tangent rotation magnitude: {}", tangent.rotation.norm());
            break;
        }

        if iteration == iterations - 1 {
            log::info!("terminating single-view optimization due to reaching maximum iterations");
            log::info!("tangent rotation magnitude: {}", tangent.rotation.norm());
            break;
        }
    }
    WorldToCamera(optimizer.pose())
}

pub fn single_view_simple_optimize_l2(
    original_pose: WorldToCamera,
    translation_trust_region: f64,
    rotation_trust_region: f64,
    iterations: usize,
    landmarks: &[FeatureWorldMatch],
) -> WorldToCamera {
    if landmarks.is_empty() {
        return original_pose;
    }
    let mut optimizer = AdaMaxSo3Tangent::new(
        original_pose.isometry(),
        translation_trust_region,
        rotation_trust_region,
        1.0,
    );
    let inv_landmark_len = (landmarks.len() as f64).recip();
    for iteration in 0..iterations {
        // Compute gradient for this sample.
        let l2 = landmarks
            .iter()
            .filter_map(|&landmark| landmark_delta(WorldToCamera(optimizer.pose()), landmark))
            .sum::<Se3TangentSpace>();

        let tangent = l2.scale(inv_landmark_len);

        if optimizer.step(tangent) {
            log::info!(
                "terminating single-view optimization due to stabilizing on iteration {}",
                iteration
            );
            log::info!("tangent rotation magnitude: {}", tangent.rotation.norm());
            break;
        }

        if iteration == iterations - 1 {
            log::info!("terminating single-view optimization due to reaching maximum iterations");
            log::info!("tangent rotation magnitude: {}", tangent.rotation.norm());
            break;
        }
    }
    WorldToCamera(optimizer.pose())
}
