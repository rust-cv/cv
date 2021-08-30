use cv_core::{
    nalgebra::{Vector2, Vector6},
    FeatureWorldMatch, Pose, Projective, Se3TangentSpace, WorldToCamera,
};
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
    mut pose: WorldToCamera,
    landmarks: &[FeatureWorldMatch],
    translation_trust_region: f64,
    rotation_trust_region: f64,
    iterations: usize,
) -> WorldToCamera {
    if landmarks.is_empty() {
        return pose;
    }
    let inv_landmark_len = (landmarks.len() as f64).recip();
    let mean_momentum = 0.9;
    let variance_momentum = 0.99;
    let epsilon = 1e-12;
    // Exponential moving averages.
    let mut ema_mean = Vector6::<f64>::zeros();
    let mut ema_variance = Vector2::<f64>::new(
        (translation_trust_region * translation_trust_region).recip(),
        (rotation_trust_region * rotation_trust_region).recip(),
    );
    let mut distances = landmarks
        .iter()
        .filter_map(|landmark| pose.transform(landmark.1).point().map(|p| p.coords.norm()))
        .collect_vec();
    distances.sort_unstable_by_key(|&v| FloatOrd(v));
    let translation_scale = distances[distances.len() / 2];
    let mut last_pose = pose;
    let mut last_rot_mag = 0.0;
    for iteration in 0..iterations {
        // Compute gradient for this sample.
        let tangent = landmarks
            .iter()
            .filter_map(|&landmark| landmark_delta(pose, landmark))
            .map(|tangent| tangent.l1())
            .sum::<Se3TangentSpace>()
            .scale(inv_landmark_len);

        let squared_norms = Vector2::new(
            tangent.translation.norm_squared()
                * (translation_trust_region * translation_trust_region).recip(),
            tangent.rotation.norm_squared()
                * (rotation_trust_region * rotation_trust_region).recip(),
        );

        // Update exponential moving average variance for this sample.
        let mut redo = false;
        ema_variance = ema_variance.zip_map(&squared_norms, |old, new| {
            let old_carry_over = variance_momentum * old;
            if new > old {
                redo = true;
                new
            } else if new > old_carry_over {
                new
            } else {
                old_carry_over
            }
        });
        // The variance is like a trust region. If it expands, we want to start over
        if redo {
            pose = last_pose;
            continue;
        }

        ema_mean = mean_momentum * ema_mean + (1.0 - mean_momentum) * tangent.to_vec();

        let mean = Se3TangentSpace::from_vec(ema_mean);
        let variance = ema_variance;
        // Compute the final scale.
        let scale = variance.map(|variance| (variance + epsilon).sqrt().recip());

        // Terminate the algorithm once it stabilizes.
        if (last_rot_mag - mean.rotation.norm()).abs() < epsilon {
            log::info!(
                "terminating single-view optimization at iteration {} due to stabilizing",
                iteration
            );
            log::info!("tangent rotation mag: {}", tangent.rotation.norm());
            break;
        }

        // Update the pose.
        let delta = mean
            .scale_translation(translation_scale * scale.x)
            .scale_rotation(scale.y);
        last_pose = pose;
        last_rot_mag = mean.rotation.norm();
        pose = WorldToCamera(delta.isometry() * pose.isometry());

        if iteration == iterations - 1 {
            log::info!("terminating single-view optimization due to reaching maximum iterations");
            log::info!("tangent rotation mag: {}", tangent.rotation.norm());
            break;
        }
    }
    pose
}
