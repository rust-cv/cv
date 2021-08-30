use crate::AdaMaxSo3Tangent;
use cv_core::{FeatureWorldMatch, Pose, Projective, Se3TangentSpace, WorldToCamera};
use cv_geom::epipolar;
use float_ord::FloatOrd;
use itertools::Itertools;

fn landmark_delta(pose: WorldToCamera, landmark: FeatureWorldMatch) -> Option<Se3TangentSpace> {
    let FeatureWorldMatch(bearing, world_point) = landmark;
    let camera_point = pose.transform(world_point);
    Some(epipolar::world_pose_gradient(
        camera_point.point()?.coords,
        bearing,
    ))
}

pub fn single_view_simple_optimize(
    original_pose: WorldToCamera,
    translation_trust_region: f64,
    rotation_trust_region: f64,
    iterations: usize,
    landmarks: &[FeatureWorldMatch],
) -> WorldToCamera {
    if landmarks.is_empty() {
        return original_pose;
    }
    let mut distances = landmarks
        .iter()
        .filter_map(|landmark| {
            original_pose
                .transform(landmark.1)
                .point()
                .map(|p| p.coords.norm())
        })
        .collect_vec();
    distances.sort_unstable_by_key(|&v| FloatOrd(v));
    let translation_scale = distances[distances.len() / 2];
    let mut optimizer = AdaMaxSo3Tangent::new(
        original_pose,
        translation_trust_region,
        rotation_trust_region,
        translation_scale,
    );
    let inv_landmark_len = (landmarks.len() as f64).recip();
    for iteration in 0..iterations {
        // Compute gradient for this sample.
        let tangent = landmarks
            .iter()
            .filter_map(|&landmark| landmark_delta(optimizer.pose(), landmark))
            .map(|tangent| tangent.l1())
            .sum::<Se3TangentSpace>()
            .scale(inv_landmark_len);

        if optimizer.step(tangent) {
            log::info!(
                "terminating single-view optimization due to stabilizing on iteration {}",
                iteration
            );
            log::info!("tangent rotation magnitude: {}", tangent.rotation.norm());
            return optimizer.pose();
        }

        if iteration == iterations - 1 {
            log::info!("terminating single-view optimization due to reaching maximum iterations");
            log::info!("tangent rotation magnitude: {}", tangent.rotation.norm());
            break;
        }
    }
    optimizer.pose()
}
