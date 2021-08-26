use cv_core::{FeatureWorldMatch, Pose, Projective, Se3TangentSpace, WorldToCamera};
use cv_geom::epipolar;

fn landmark_delta(pose: WorldToCamera, landmark: FeatureWorldMatch) -> Se3TangentSpace {
    let FeatureWorldMatch(bearing, world_point) = landmark;
    let camera_point = pose.transform(world_point);
    epipolar::relative_pose_gradient(
        pose.isometry().translation.vector,
        camera_point.bearing(),
        bearing,
    )
}

// fn landmark_delta(pose: WorldToCamera, landmark: FeatureWorldMatch) -> Option<Se3TangentSpace> {
//     let FeatureWorldMatch(bearing, world_point) = landmark;
//     let camera_point = pose.transform(world_point);
//     Some(epipolar::world_pose_gradient(
//         camera_point.point()?.coords,
//         bearing,
//     ))
// }

pub fn single_view_simple_optimize(
    mut pose: WorldToCamera,
    landmarks: &[FeatureWorldMatch],
    optimization_rate: f64,
    iterations: usize,
) -> WorldToCamera {
    let scale = optimization_rate / landmarks.len() as f64;
    for _ in 0..iterations {
        let net_delta: Se3TangentSpace = landmarks
            .iter()
            .map(|&landmark| landmark_delta(pose, landmark))
            .sum();
        pose.0 = net_delta.scale(scale).isometry() * pose.0;
    }
    pose
}
