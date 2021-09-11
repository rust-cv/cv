use crate::AdaMaxSo3Tangent;
use cv_core::{
    nalgebra::{IsometryMatrix3, UnitVector3},
    CameraToCamera, Pose, Se3TangentSpace,
};
use cv_geom::epipolar;
use float_ord::FloatOrd;
use itertools::izip;

fn landmark_deltas(
    poses: [IsometryMatrix3<f64>; 2],
    observations: [UnitVector3<f64>; 3],
) -> [Se3TangentSpace; 2] {
    let ftoc = poses[0];
    let stoc = poses[1];

    epipolar::three_view_gradients(
        observations[0],
        ftoc * observations[1],
        ftoc.translation.vector,
        stoc * observations[2],
        stoc.translation.vector,
    )
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
    let mut poses = poses.map(|pose| pose.isometry().inverse());
    let mut prev_deltas = [Se3TangentSpace::identity(), Se3TangentSpace::identity()];
    for iteration in 0..iterations {
        // For the sums here, we use a VERY small number.
        // This is so that if the gradient is zero for every data point (we are 100% perfect),
        // then it will not turn the delta into NaN by taking the reciprocal of 0 (infinity) and multiplying
        // it by 0.
        let mut nets = [
            (Se3TangentSpace::identity(), 0.0, 0.0),
            (Se3TangentSpace::identity(), 0.0, 0.0),
        ];
        let tscale = poses
            .iter()
            .map(|pose| pose.translation.vector.norm())
            .sum::<f64>();
        for &observations in landmarks {
            let deltas = landmark_deltas(poses, observations);

            for ((l1sum, ts, rs), &delta) in nets.iter_mut().zip(deltas.iter()) {
                *ts += (delta.translation.norm() + tscale * 1e-9).recip();
                *rs += (delta.rotation.norm() + 1e-9).recip();
                *l1sum += delta.l1();
            }
        }

        // Apply the harmonic mean as per the Weiszfeld algorithm from the paper
        // "Sur le point pour lequel la somme des distances de n points donn ́es est minimum."
        let mut deltas = [Se3TangentSpace::identity(); 2];
        for (delta, (l1sum, ts, rs)) in deltas.iter_mut().zip(nets) {
            *delta = l1sum
                .scale_translation(ts.recip())
                .scale_rotation(rs.recip());
        }

        // Check if all of the optimizers reached stability.
        let done = prev_deltas == deltas;
        if done {
            log::info!(
                "terminating three-view optimization due to stabilizing on iteration {}",
                iteration
            );
            log::info!(
                "first rotation magnitude: {}",
                nets[0].0.rotation.norm() / landmarks.len() as f64
            );
            log::info!(
                "second rotation magnitude: {}",
                nets[1].0.rotation.norm() / landmarks.len() as f64
            );
            break;
        }

        // Run everything through the optimizer and keep track of if all of them finished.
        for (pose, delta) in poses.iter_mut().zip(deltas.iter()) {
            // Perturb slightly using the previous delta to avoid getting stuck overlapping with a datapoint.
            // This can occur to the Weizfeld algorithm because when you overlap a datapoint perfectly, it produces a 0.
            // The 0 ends up causing the whole harmonic mean to go to 0. This small perturbation helps avoid
            // getting stuck in a local minima.
            *pose = delta.isometry() * *pose;
        }

        prev_deltas = deltas;

        // If we are on the last iteration, print some logs indicating so.
        if iteration == iterations - 1 {
            log::info!("terminating three-view optimization due to reaching maximum iterations");
            log::info!(
                "first rotation magnitude: {}",
                nets[0].0.rotation.norm() / landmarks.len() as f64
            );
            log::info!(
                "second rotation magnitude: {}",
                nets[1].0.rotation.norm() / landmarks.len() as f64
            );
            break;
        }
    }
    poses.map(|pose| pose.inverse().into())
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
            pose.isometry().inverse(),
            translation_trust_region,
            rotation_trust_region,
        )
    });
    let inv_landmark_len = (landmarks.len() as f64).recip();
    for iteration in 0..iterations {
        let poses = optimizers.clone().map(|o| o.pose());
        // Extract the tangent vectors for each pose.
        let mut net_l2_deltas = [Se3TangentSpace::identity(); 2];
        for &observations in landmarks {
            let deltas = landmark_deltas(poses, observations);

            for (net_l2, &delta) in net_l2_deltas.iter_mut().zip(deltas.iter()) {
                *net_l2 += delta.scale(inv_landmark_len);
            }
        }

        let net_deltas = net_l2_deltas;

        // Run everything through the optimizer and keep track of if all of them finished.
        let mut done = true;
        for (optimizer, net_delta) in optimizers
            .iter_mut()
            .zip(std::array::IntoIter::new(net_deltas))
        {
            if !optimizer.step_linear_translation(net_delta) {
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
    optimizers.map(|optimizer| CameraToCamera(optimizer.pose().inverse()))
}
