use cv_core::{FeatureWorldMatch, Pose, Projective, Se3TangentSpace, WorldToCamera};
use cv_geom::epipolar;

pub(crate) fn landmark_delta(
    pose: WorldToCamera,
    landmark: FeatureWorldMatch,
) -> Option<Se3TangentSpace> {
    let FeatureWorldMatch(bearing, world_point) = landmark;
    let camera_point = pose.transform(world_point);
    Some(epipolar::world_pose_gradient(
        camera_point.point()?.coords,
        bearing,
    ))
}

pub fn single_view_simple_optimize_l1(
    mut pose: WorldToCamera,
    epsilon: f64,
    optimization_rate: f64,
    iterations: usize,
    landmarks: &[FeatureWorldMatch],
) -> WorldToCamera {
    if landmarks.is_empty() {
        return pose;
    }
    let mut best_trans = f64::INFINITY;
    let mut best_rot = f64::INFINITY;
    let mut no_improve_for = 0;
    for iteration in 0..iterations {
        let tscale = pose.isometry().translation.vector.norm();
        let mut l1sum = Se3TangentSpace::identity();
        let mut ts = 0.0;
        let mut rs = 0.0;
        for &landmark in landmarks {
            if let Some(tangent) = landmark_delta(pose, landmark) {
                ts += (tangent.translation.norm() + tscale * epsilon).recip();
                rs += (tangent.rotation.norm() + epsilon).recip();
                l1sum += tangent.l1();
            }
        }

        let delta = l1sum
            .scale(optimization_rate)
            .scale_translation(ts.recip())
            .scale_rotation(rs.recip());

        no_improve_for += 1;
        let t = l1sum.translation.norm();
        let r = l1sum.rotation.norm();
        if best_trans > t {
            best_trans = t;
            no_improve_for = 0;
        }
        if best_rot > r {
            best_rot = r;
            no_improve_for = 0;
        }

        if no_improve_for >= 50 {
            log::info!(
                "terminating single-view optimization due to stabilizing on iteration {}",
                iteration
            );
            log::info!("tangent rotation magnitude: {}", l1sum.rotation.norm());
            break;
        }

        // Update the pose.
        pose = (delta.isometry() * pose.isometry()).into();

        if iteration == iterations - 1 {
            log::info!("terminating single-view optimization due to reaching maximum iterations");
            log::info!("tangent rotation magnitude: {}", l1sum.rotation.norm());
            break;
        }
    }
    pose
}

pub fn single_view_simple_optimize_l2(
    mut pose: WorldToCamera,
    optimization_rate: f64,
    iterations: usize,
    landmarks: &[FeatureWorldMatch],
) -> WorldToCamera {
    if landmarks.is_empty() {
        return pose;
    }
    let mut best_trans = f64::INFINITY;
    let mut best_rot = f64::INFINITY;
    let mut no_improve_for = 0;
    let inv_landmark_len = (landmarks.len() as f64).recip();
    for iteration in 0..iterations {
        let mut l2sum = Se3TangentSpace::identity();
        for &landmark in landmarks {
            if let Some(tangent) = landmark_delta(pose, landmark) {
                l2sum += tangent;
            }
        }

        let tangent = l2sum.scale(inv_landmark_len);
        let delta = tangent.scale(optimization_rate);

        no_improve_for += 1;
        let t = l2sum.translation.norm();
        let r = l2sum.rotation.norm();
        if best_trans > t {
            best_trans = t;
            no_improve_for = 0;
        }
        if best_rot > r {
            best_rot = r;
            no_improve_for = 0;
        }

        if no_improve_for >= 50 {
            log::info!(
                "terminating single-view optimization due to stabilizing on iteration {}",
                iteration
            );
            log::info!("tangent rotation magnitude: {}", tangent.rotation.norm());
            break;
        }

        // Update the pose.
        pose = (delta.isometry() * pose.isometry()).into();

        if iteration == iterations - 1 {
            log::info!("terminating single-view optimization due to reaching maximum iterations");
            log::info!("tangent rotation magnitude: {}", tangent.rotation.norm());
            break;
        }
    }
    pose
}
