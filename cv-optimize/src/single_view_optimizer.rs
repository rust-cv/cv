use crate::AdaMaxSo3Tangent;
use cv_core::{FeatureWorldMatch, Pose, Projective, Se3TangentSpace, WorldToCamera};
use cv_geom::epipolar;

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
    for iteration in 0..iterations {
        // Compute gradient for this sample.
        let l2 = landmarks
            .iter()
            .filter_map(|&landmark| landmark_delta(WorldToCamera(optimizer.pose()), landmark))
            .sum::<Se3TangentSpace>();
        let l1 = landmarks
            .iter()
            .filter_map(|&landmark| landmark_delta(WorldToCamera(optimizer.pose()), landmark))
            .map(|tangent| tangent.l1())
            .sum::<Se3TangentSpace>();

        let tangent = l1
            .scale_translation(l2.translation.norm())
            .scale_rotation(l2.rotation.norm())
            .scale(inv_landmark_len);

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
