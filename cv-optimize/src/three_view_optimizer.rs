use crate::{epipolar_gradient, Se3TangentSpace};
use cv_core::{nalgebra::UnitVector3, CameraToCamera, Pose};

fn landmark_deltas(
    poses: [CameraToCamera; 2],
    observations: [UnitVector3<f64>; 3],
) -> [Se3TangentSpace; 3] {
    let ctof = poses[0].isometry();
    let ftoc = ctof.inverse();
    let ctos = poses[1].isometry();
    let stoc = ctos.inverse();
    let ftos = ctos * ftoc;
    let stof = ftos.inverse();

    [
        epipolar_gradient(
            ftoc.translation.vector,
            ftoc * observations[1],
            observations[0],
        ) + epipolar_gradient(
            stoc.translation.vector,
            stoc * observations[2],
            observations[0],
        ),
        epipolar_gradient(
            ctof.translation.vector,
            ctof * observations[0],
            observations[1],
        ) + epipolar_gradient(
            stof.translation.vector,
            stof * observations[2],
            observations[1],
        ),
        epipolar_gradient(
            ctos.translation.vector,
            ctos * observations[0],
            observations[2],
        ) + epipolar_gradient(
            ftos.translation.vector,
            ftos * observations[1],
            observations[2],
        ),
    ]
}

pub fn three_view_simple_optimize(
    mut poses: [CameraToCamera; 2],
    landmarks: &[[UnitVector3<f64>; 3]],
    optimization_rate: f64,
    iterations: usize,
) -> [CameraToCamera; 2] {
    for _ in 0..iterations {
        let mut net_deltas = [Se3TangentSpace::identity(); 3];
        for &observations in landmarks {
            let deltas = landmark_deltas(poses, observations);
            for (net, &delta) in net_deltas.iter_mut().zip(deltas.iter()) {
                *net = *net + delta;
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
