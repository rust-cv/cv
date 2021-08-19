use cv_core::{FeatureWorldMatch, Pose, Projective, WorldToCamera};

use crate::{observation_gradient, Se3TangentSpace};

fn landmark_delta(pose: WorldToCamera, landmark: FeatureWorldMatch) -> Option<Se3TangentSpace> {
    let FeatureWorldMatch(bearing, world_point) = landmark;
    let camera_point = pose.transform(world_point).point()?;
    Some(observation_gradient(camera_point, bearing))
}

pub fn single_view_simple_optimize(
    mut pose: WorldToCamera,
    landmarks: &[FeatureWorldMatch],
    optimization_rate: f64,
    iterations: usize,
) -> WorldToCamera {
    for _ in 0..iterations {
        let mut net_delta = Se3TangentSpace::identity();
        for &landmark in landmarks {
            if let Some(delta) = landmark_delta(pose, landmark) {
                net_delta = net_delta + delta;
            }
        }
        let scale = optimization_rate / landmarks.len() as f64;
        pose = WorldToCamera(net_delta.scale(scale).isometry() * pose.isometry());
    }
    pose
}
