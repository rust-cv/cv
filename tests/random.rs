use cv_core::nalgebra::{IsometryMatrix3, Rotation3, Vector2, Vector3};
use cv_core::sample_consensus::{Estimator, Model};
use cv_core::{geom, CameraPoint, KeyPointsMatch, NormalizedKeyPoint, RelativeCameraPose};

const SAMPLE_POINTS: usize = 16;
const RESIDUAL_THRESHOLD: f64 = 1e-4;

const ROT_MAGNITUDE: f64 = 0.1;
const POINT_BOX_SIZE: f64 = 2.0;
const POINT_DISTANCE: f64 = 1.0;

#[test]
fn randomized() {
    let successes = (0..1000).filter(|_| run_round()).count();
    eprintln!("successes: {}", successes);
    assert!(successes > 950);
}

fn run_round() -> bool {
    let mut success = true;
    let (real_pose, aps, bps, depths) = some_test_data();
    let matches = aps.iter().zip(&bps).map(|(&a, &b)| KeyPointsMatch(a, b));
    let eight_point = eight_point::EightPoint::new();
    let essential = eight_point
        .estimate(matches.clone())
        .expect("didn't get any essential matrix");
    for m in matches.clone() {
        if essential.residual(&m).abs() > RESIDUAL_THRESHOLD as f32 {
            success = false;
            eprintln!("failed residual check: {}", essential.residual(&m).abs());
        }
    }

    // Get the possible poses for the essential matrix created from `pose`.
    let estimate_pose = match essential.solve_pose(
        1e-6,
        50,
        0.1,
        geom::triangulate_bearing_reproject,
        matches
            .zip(depths)
            .map(|(KeyPointsMatch(a, b), depth)| (a.with_depth(depth), b)),
    ) {
        Some(pose) => pose,
        None => {
            eprintln!("estimation failure angle: {}", real_pose.rotation.angle());
            return false;
        }
    };

    let rot_axis_residual = (1.0
        - estimate_pose
            .rotation
            .axis()
            .unwrap()
            .dot(&real_pose.rotation.axis().unwrap()))
    .abs();
    let rot_angle_residual = (estimate_pose.rotation.angle() - real_pose.rotation.angle()).abs();
    let translation_residual =
        (real_pose.translation.vector - estimate_pose.translation.vector).norm();
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
    RelativeCameraPose,
    [NormalizedKeyPoint; SAMPLE_POINTS],
    [NormalizedKeyPoint; SAMPLE_POINTS],
    impl Iterator<Item = f64> + Clone,
) {
    // The relative pose orientation is fixed and translation is random.
    let relative_pose = RelativeCameraPose(IsometryMatrix3::from_parts(
        Vector3::new_random().into(),
        Rotation3::new(Vector3::new_random() * std::f64::consts::PI * 2.0 * ROT_MAGNITUDE),
    ));

    // Generate A's camera points.
    let cams_a = (0..SAMPLE_POINTS)
        .map(|_| {
            let mut a = Vector3::new_random() * POINT_BOX_SIZE;
            a.x -= 0.5 * POINT_BOX_SIZE;
            a.y -= 0.5 * POINT_BOX_SIZE;
            a.z += 1.0 + POINT_DISTANCE;
            CameraPoint(a.into())
        })
        .collect::<Vec<_>>()
        .into_iter();

    // Generate B's camera points.
    let cams_b = cams_a.clone().map(|a| relative_pose.transform(a));

    let mut kps_a = [NormalizedKeyPoint(Vector2::zeros().into()); SAMPLE_POINTS];
    for (keypoint, camera) in kps_a.iter_mut().zip(cams_a.clone()) {
        *keypoint = camera.into();
    }
    let mut kps_b = [NormalizedKeyPoint(Vector2::zeros().into()); SAMPLE_POINTS];
    for (keypoint, camera) in kps_b.iter_mut().zip(cams_b.clone()) {
        *keypoint = camera.into();
    }

    (relative_pose, kps_a, kps_b, cams_a.map(|p| p.0.z))
}
