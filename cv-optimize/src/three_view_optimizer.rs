use crate::AdaMaxSo3Tangent;
use cv_core::{nalgebra::UnitVector3, CameraToCamera, Pose, Se3TangentSpace};
use cv_geom::epipolar;
use float_ord::FloatOrd;
use itertools::izip;

fn landmark_deltas(
    poses: [CameraToCamera; 2],
    observations: [UnitVector3<f64>; 3],
) -> [Se3TangentSpace; 2] {
    let ctof = poses[0].isometry();
    let ftoc = ctof.inverse();
    let ctos = poses[1].isometry();
    let ftos = ctos * ftoc;
    let stof = ftos.inverse();

    [
        epipolar::relative_pose_gradient(
            ctof.translation.vector,
            ctof * observations[0],
            observations[1],
        ) + epipolar::relative_pose_gradient(
            stof.translation.vector,
            stof * observations[2],
            observations[1],
        )
        .scale(0.5),
        epipolar::relative_pose_gradient(
            ctos.translation.vector,
            ctos * observations[0],
            observations[2],
        ) + epipolar::relative_pose_gradient(
            ftos.translation.vector,
            ftos * observations[1],
            observations[2],
        )
        .scale(0.5),
    ]
}

pub fn three_view_simple_optimize_l1(
    poses: [CameraToCamera; 2],
    translation_trust_region: f64,
    rotation_trust_region: f64,
    iterations: usize,
    landmarks: &[[UnitVector3<f64>; 3]],
) -> [CameraToCamera; 2] {
    if landmarks.is_empty() {
        return poses;
    }
    let mut optimizers = poses.map(|pose| {
        AdaMaxSo3Tangent::new(
            pose.isometry(),
            translation_trust_region,
            rotation_trust_region,
            1.0,
        )
    });
    let inv_landmark_len = (landmarks.len() as f64).recip();
    let mut ts = [vec![0.0; landmarks.len()], vec![0.0; landmarks.len()]];
    let mut rs = [vec![0.0; landmarks.len()], vec![0.0; landmarks.len()]];
    for iteration in 0..iterations {
        // Extract the tangent vectors for each pose.
        for v in ts.iter_mut().chain(rs.iter_mut()) {
            v.clear();
        }
        let mut net_l1s = [Se3TangentSpace::identity(), Se3TangentSpace::identity()];
        for &observations in landmarks {
            let deltas = landmark_deltas(poses, observations);

            for ((l1sum, ts, rs), &delta) in
                izip!(net_l1s.iter_mut(), ts.iter_mut(), rs.iter_mut()).zip(deltas.iter())
            {
                ts.push(delta.translation.norm());
                rs.push(delta.rotation.norm());
                *l1sum += delta.l1();
            }
        }
        for v in ts.iter_mut().chain(rs.iter_mut()) {
            v.sort_unstable_by_key(|&f| FloatOrd(f));
        }

        let mut net_deltas = [Se3TangentSpace::identity(); 2];
        for (delta, (l1sum, ts, rs)) in
            net_deltas
                .iter_mut()
                .zip(izip!(net_l1s.iter(), ts.iter(), rs.iter()))
        {
            let translation_scale = ts[ts.len() / 2] * inv_landmark_len;
            let rotation_scale = rs[rs.len() / 2] * inv_landmark_len;
            *delta = l1sum
                .scale(inv_landmark_len)
                .scale_translation(translation_scale)
                .scale_rotation(rotation_scale);
        }

        // Run everything through the optimizer and keep track of if all of them finished.
        let mut done = true;
        for (optimizer, net_delta) in optimizers
            .iter_mut()
            .zip(std::array::IntoIter::new(net_deltas))
        {
            if !optimizer.step(net_delta) {
                done = false;
            }
        }

        // Check if all of the optimizers reached stability.
        if done {
            log::info!(
                "terminating single-view optimization due to stabilizing on iteration {}",
                iteration
            );
            log::info!(
                "first rotation magnitude: {}",
                net_deltas[0].rotation.norm()
            );
            log::info!(
                "second rotation magnitude: {}",
                net_deltas[1].rotation.norm()
            );
            break;
        }

        // If we are on the last iteration, print some logs indicating so.
        if iteration == iterations - 1 {
            log::info!("terminating single-view optimization due to reaching maximum iterations");
            log::info!(
                "first rotation magnitude: {}",
                net_deltas[0].rotation.norm()
            );
            log::info!(
                "second rotation magnitude: {}",
                net_deltas[1].rotation.norm()
            );
            break;
        }
    }
    optimizers.map(|optimizer| CameraToCamera(optimizer.pose()))
}

pub fn three_view_simple_optimize_l2(
    poses: [CameraToCamera; 2],
    translation_trust_region: f64,
    rotation_trust_region: f64,
    iterations: usize,
    landmarks: &[[UnitVector3<f64>; 3]],
) -> [CameraToCamera; 2] {
    if landmarks.is_empty() {
        return poses;
    }
    let mut optimizers = poses.map(|pose| {
        AdaMaxSo3Tangent::new(
            pose.isometry(),
            translation_trust_region,
            rotation_trust_region,
            1.0,
        )
    });
    let inv_landmark_len = (landmarks.len() as f64).recip();
    for iteration in 0..iterations {
        // Extract the tangent vectors for each pose.
        let mut net_l2_deltas = [Se3TangentSpace::identity(); 2];
        for &observations in landmarks {
            let deltas = landmark_deltas(poses, observations);

            for (net_l2, &delta) in net_l2_deltas.iter_mut().zip(deltas.iter()) {
                *net_l2 += delta;
            }
        }

        let mut net_deltas = net_l2_deltas;
        for delta in &mut net_deltas {
            *delta = delta.scale(inv_landmark_len);
        }

        // Run everything through the optimizer and keep track of if all of them finished.
        let mut done = true;
        for (optimizer, net_delta) in optimizers
            .iter_mut()
            .zip(std::array::IntoIter::new(net_deltas))
        {
            if !optimizer.step(net_delta) {
                done = false;
            }
        }

        // Check if all of the optimizers reached stability.
        if done {
            log::info!(
                "terminating three-view optimization due to stabilizing on iteration {}",
                iteration
            );
            log::info!(
                "first rotation magnitude: {}",
                net_deltas[0].rotation.norm()
            );
            log::info!(
                "second rotation magnitude: {}",
                net_deltas[1].rotation.norm()
            );
            break;
        }

        // If we are on the last iteration, print some logs indicating so.
        if iteration == iterations - 1 {
            log::info!("terminating three-view optimization due to reaching maximum iterations");
            log::info!(
                "first rotation magnitude: {}",
                net_deltas[0].rotation.norm()
            );
            log::info!(
                "second rotation magnitude: {}",
                net_deltas[1].rotation.norm()
            );
            break;
        }
    }
    optimizers.map(|optimizer| CameraToCamera(optimizer.pose()))
}
