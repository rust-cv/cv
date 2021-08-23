use cv_core::{FeatureWorldMatch, Pose, Projective, WorldToCamera};
use nalgebra::Vector3;

use crate::point_translation_gradient;

fn landmark_delta(pose: WorldToCamera, landmark: FeatureWorldMatch) -> Option<Vector3<f64>> {
    let FeatureWorldMatch(bearing, world_point) = landmark;
    let camera_point = pose.transform(world_point).point()?;
    Some(point_translation_gradient(camera_point, bearing))
}

pub fn single_view_simple_optimize(
    mut pose: WorldToCamera,
    landmarks: &[FeatureWorldMatch],
    optimization_rate: f64,
    iterations: usize,
) -> WorldToCamera {
    for _ in 0..iterations {
        let mut net_translation = Vector3::zeros();
        for &landmark in landmarks {
            if let Some(delta) = landmark_delta(pose, landmark) {
                net_translation += delta;
            }
        }
        let scale = optimization_rate / landmarks.len() as f64;
        pose.0
            .append_translation_mut(&(scale * net_translation).into());
    }
    pose
}
