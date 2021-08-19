use crate::{observation_gradient, Se3TangentSpace};
use cv_core::{nalgebra::UnitVector3, CameraToCamera, Pose, Projective, TriangulatorObservations};

fn landmark_deltas(
    poses: [CameraToCamera; 2],
    observations: [UnitVector3<f64>; 3],
    triangulator: &impl TriangulatorObservations,
) -> Option<[Se3TangentSpace; 3]> {
    let center_point = triangulator
        .triangulate_observations_to_camera(
            observations[0],
            poses.iter().copied().zip(observations[1..].iter().copied()),
        )?
        .point()?;
    let first_point = poses[0].isometry().transform_point(&center_point);
    let second_point = poses[1].isometry().transform_point(&center_point);

    Some(
        [
            (center_point, observations[0]),
            (first_point, observations[1]),
            (second_point, observations[2]),
        ]
        .map(|(point, bearing)| observation_gradient(point, bearing)),
    )
}

pub fn three_view_simple_optimize(
    mut poses: [CameraToCamera; 2],
    triangulator: &impl TriangulatorObservations,
    landmarks: &[[UnitVector3<f64>; 3]],
    optimization_rate: f64,
    iterations: usize,
) -> [CameraToCamera; 2] {
    for _ in 0..iterations {
        let mut net_deltas = [Se3TangentSpace::identity(); 3];
        for &observations in landmarks {
            if let Some(deltas) = landmark_deltas(poses, observations, triangulator) {
                for (net, &delta) in net_deltas.iter_mut().zip(deltas.iter()) {
                    *net = *net + delta;
                }
            }
        }
        let scale = optimization_rate / landmarks.len() as f64;
        for (pose, &net_delta) in poses.iter_mut().zip(net_deltas[1..].iter()) {
            *pose = CameraToCamera(
                net_delta.scale(scale).isometry()
                    * pose.isometry()
                    * net_deltas[0].scale(scale).isometry().inverse(),
            );
        }
    }
    poses
}
