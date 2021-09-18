use cv_core::{
    nalgebra::{IsometryMatrix3, UnitVector3},
    CameraToCamera, Pose, Se3TangentSpace,
};
use cv_geom::epipolar;

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
    epsilon: f64,
    optimization_rate: f64,
    iterations: usize,
    landmarks: &[[UnitVector3<f64>; 3]],
) -> [CameraToCamera; 2] {
    if landmarks.is_empty() {
        return poses;
    }
    let mut poses = poses.map(|pose| pose.isometry().inverse());
    let mut bests = [[f64::INFINITY; 2]; 2];
    let mut no_improve_for = 0;
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
                *ts += (delta.translation.norm() + tscale * epsilon).recip();
                *rs += (delta.rotation.norm() + epsilon).recip();
                *l1sum += delta.l1();
            }
        }

        // Apply the harmonic mean as per the Weiszfeld algorithm from the paper
        // "Sur le point pour lequel la somme des distances de n points donn ́es est minimum."
        let mut deltas = [Se3TangentSpace::identity(); 2];
        for (delta, (l1sum, ts, rs)) in deltas.iter_mut().zip(nets) {
            *delta = l1sum
                .scale(optimization_rate)
                .scale_translation(ts.recip())
                .scale_rotation(rs.recip());
        }

        no_improve_for += 1;
        for ([best_t, best_r], (l1, _, _)) in bests.iter_mut().zip(nets.iter()) {
            let t = l1.translation.norm();
            let r = l1.rotation.norm();
            if *best_t > t {
                *best_t = t;
                no_improve_for = 0;
            }
            if *best_r > r {
                *best_r = r;
                no_improve_for = 0;
            }
        }

        // Check if all of the optimizers reached stability.
        if no_improve_for >= 50 {
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
    optimization_rate: f64,
    iterations: usize,
    landmarks: &[[UnitVector3<f64>; 3]],
) -> [CameraToCamera; 2] {
    if landmarks.is_empty() {
        return poses;
    }
    let inv_landmark_len = (landmarks.len() as f64).recip();
    let mut poses = poses.map(|pose| pose.isometry().inverse());
    let mut bests = [[f64::INFINITY; 2]; 2];
    let mut no_improve_for = 0;
    for iteration in 0..iterations {
        // For the sums here, we use a VERY small number.
        // This is so that if the gradient is zero for every data point (we are 100% perfect),
        // then it will not turn the delta into NaN by taking the reciprocal of 0 (infinity) and multiplying
        // it by 0.
        let mut nets = [Se3TangentSpace::identity(), Se3TangentSpace::identity()];
        for &observations in landmarks {
            let deltas = landmark_deltas(poses, observations);

            for (l2sum, &delta) in nets.iter_mut().zip(deltas.iter()) {
                *l2sum += delta;
            }
        }

        // Apply the harmonic mean as per the Weiszfeld algorithm from the paper
        // "Sur le point pour lequel la somme des distances de n points donn ́es est minimum."
        let mut deltas = [Se3TangentSpace::identity(); 2];
        for (delta, l2sum) in deltas.iter_mut().zip(nets) {
            *delta = l2sum.scale(inv_landmark_len * optimization_rate);
        }

        no_improve_for += 1;
        for ([best_t, best_r], l2sum) in bests.iter_mut().zip(nets.iter()) {
            let t = l2sum.translation.norm();
            let r = l2sum.rotation.norm();
            if *best_t > t {
                *best_t = t;
                no_improve_for = 0;
            }
            if *best_r > r {
                *best_r = r;
                no_improve_for = 0;
            }
        }

        // Check if all of the optimizers reached stability.
        if no_improve_for >= 50 {
            log::info!(
                "terminating three-view optimization due to stabilizing on iteration {}",
                iteration
            );
            log::info!("first rotation magnitude: {}", deltas[0].rotation.norm());
            log::info!("second rotation magnitude: {}", deltas[1].rotation.norm());
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

        // If we are on the last iteration, print some logs indicating so.
        if iteration == iterations - 1 {
            log::info!("terminating three-view optimization due to reaching maximum iterations");
            log::info!("first rotation magnitude: {}", deltas[0].rotation.norm());
            log::info!("second rotation magnitude: {}", deltas[1].rotation.norm());
            break;
        }
    }
    poses.map(|pose| pose.inverse().into())
}
