use cv_geom::epipolar_gradient;
use cv_core::{FeatureWorldMatch, Pose, Projective, Se3TangentSpace, WorldToCamera};

fn landmark_delta(pose: WorldToCamera, landmark: FeatureWorldMatch) -> Se3TangentSpace {
    let FeatureWorldMatch(bearing, world_point) = landmark;
    let camera_point = pose.transform(world_point);
    epipolar_gradient(
        pose.isometry().translation.vector,
        camera_point.bearing(),
        bearing,
    )
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
            net_delta += landmark_delta(pose, landmark);
        }
        let scale = optimization_rate / landmarks.len() as f64;
        pose.0 = net_delta.scale(scale).isometry() * pose.0;
    }
    pose
}
