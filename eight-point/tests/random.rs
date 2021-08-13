use cv_core::{
    nalgebra::{IsometryMatrix3, Point3, Rotation3, UnitVector3, Vector3},
    sample_consensus::{Estimator, Model},
    CameraPoint, CameraToCamera, FeatureMatch, Pose, Projective,
};

const SAMPLE_POINTS: usize = 16;
const RESIDUAL_THRESHOLD: f64 = 1e-4;

const ROT_MAGNITUDE: f64 = 0.2;
const POINT_BOX_SIZE: f64 = 2.0;
const POINT_DISTANCE: f64 = 3.0;

#[test]
fn randomized() {
    let successes = (0..1000).filter(|_| run_round()).count();
    eprintln!("successes: {}", successes);
    assert!(successes > 950);
}

fn run_round() -> bool {
    let mut success = true;
    let (real_pose, aps, bps) = some_test_data();
    let matches = aps.iter().zip(&bps).map(|(&a, &b)| FeatureMatch(a, b));
    let eight_point = eight_point::EightPoint::new();
    let essential = eight_point
        .estimate(matches.clone())
        .expect("didn't get any essential matrix");
    for m in matches.clone() {
        if essential.residual(&m).abs() > RESIDUAL_THRESHOLD {
            success = false;
            eprintln!("failed residual check: {}", essential.residual(&m).abs());
        }
    }

    // Get the possible poses for the essential matrix created from `pose`.
    let estimate_pose = match essential.pose_solver().solve_unscaled(matches) {
        Some(pose) => pose,
        None => {
            return false;
        }
    };

    let rot_axis_residual = 1.0
        - estimate_pose
            .0
            .rotation
            .axis()
            .unwrap()
            .dot(&real_pose.0.rotation.axis().unwrap());
    let rot_angle_residual =
        (estimate_pose.0.rotation.angle() - real_pose.0.rotation.angle()).abs();
    let translation_residual = 1.0
        - real_pose
            .0
            .translation
            .vector
            .normalize()
            .dot(&estimate_pose.0.translation.vector.normalize());
    success &= rot_axis_residual < RESIDUAL_THRESHOLD;
    success &= rot_angle_residual < RESIDUAL_THRESHOLD;
    success &= translation_residual < RESIDUAL_THRESHOLD;
    if !success {
        eprintln!("rot angle residual({})", rot_angle_residual);
        eprintln!("rot axis residual({})", rot_axis_residual);
        eprintln!("translation residual({})", translation_residual);
        eprintln!("real pose: {:?}", real_pose);
        eprintln!("estimate pose: {:?}", estimate_pose);
    }
    success
}

/// Gets a random relative pose, input points A, input points B, and A point depths.
fn some_test_data() -> (
    CameraToCamera,
    [UnitVector3<f64>; SAMPLE_POINTS],
    [UnitVector3<f64>; SAMPLE_POINTS],
) {
    // The relative pose orientation is fixed and translation is random.
    let relative_pose = CameraToCamera(IsometryMatrix3::from_parts(
        Vector3::new_random().into(),
        Rotation3::new(Vector3::new_random() * std::f64::consts::PI * 2.0 * ROT_MAGNITUDE),
    ));

    // Generate A's camera points.
    let cams_a = (0..SAMPLE_POINTS)
        .map(|_| {
            let mut a = Point3::from(Vector3::new_random() * POINT_BOX_SIZE);
            a.x -= 0.5 * POINT_BOX_SIZE;
            a.y -= 0.5 * POINT_BOX_SIZE;
            a.z += POINT_DISTANCE;
            CameraPoint::from_point(a)
        })
        .collect::<Vec<_>>()
        .into_iter();

    // Generate B's camera points.
    let cams_b = cams_a.clone().map(|a| relative_pose.transform(a));

    let mut kps_a = [UnitVector3::new_normalize(Vector3::z()); SAMPLE_POINTS];
    for (keypoint, camera) in kps_a.iter_mut().zip(cams_a) {
        *keypoint = camera.bearing();
    }
    let mut kps_b = [UnitVector3::new_normalize(Vector3::z()); SAMPLE_POINTS];
    for (keypoint, camera) in kps_b.iter_mut().zip(cams_b.clone()) {
        *keypoint = camera.bearing();
    }

    (relative_pose, kps_a, kps_b)
}
