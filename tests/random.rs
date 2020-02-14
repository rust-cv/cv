use cv_core::nalgebra::{Isometry3, UnitQuaternion, Vector2, Vector3};
use cv_core::sample_consensus::{Estimator, Model};
use cv_core::{geom, CameraPoint, KeyPointsMatch, NormalizedKeyPoint, RelativeCameraPose};

const SAMPLE_POINTS: usize = 16;
const RESIDUAL_THRESHOLD: f64 = 1e-4;

const ROT_MAGNITUDE: f64 = 0.5;
const POINT_BOX_SIZE: f64 = 2.0;

#[test]
fn randomized() {
    let (real_pose, aps, bps, depths) = some_test_data();
    let matches = aps.iter().zip(&bps).map(|(&a, &b)| KeyPointsMatch(a, b));
    let eight_point = eight_point::EightPoint::new();
    let essential = eight_point
        .estimate(matches.clone())
        .expect("didn't get any essential matrix");
    for m in matches.clone() {
        assert!(essential.residual(&m).abs() < RESIDUAL_THRESHOLD as f32);
    }

    // Get the possible poses for the essential matrix created from `pose`.
    let estimate_pose = essential
        .solve_pose(
            0.1,
            1e-6,
            50,
            geom::triangulate_bearing_intersection,
            matches
                .zip(depths)
                .map(|(KeyPointsMatch(a, b), depth)| (a.with_depth(depth), b)),
        )
        .unwrap();

    let angle_residual = estimate_pose
        .rotation
        .rotation_to(&real_pose.rotation)
        .angle();
    let translation_residual =
        (real_pose.translation.vector - estimate_pose.translation.vector).norm();
    assert!(
        angle_residual < RESIDUAL_THRESHOLD && translation_residual < RESIDUAL_THRESHOLD,
        "angle({}), translation({})",
        angle_residual,
        translation_residual
    );
}

/// Gets a random relative pose, input points A, input points B, and A point depths.
fn some_test_data() -> (
    RelativeCameraPose,
    [NormalizedKeyPoint; SAMPLE_POINTS],
    [NormalizedKeyPoint; SAMPLE_POINTS],
    impl Iterator<Item = f64> + Clone,
) {
    // The relative pose orientation is fixed and translation is random.
    let relative_pose = RelativeCameraPose(Isometry3::from_parts(
        Vector3::new_random().into(),
        UnitQuaternion::new(Vector3::new_random() * std::f64::consts::PI * 2.0 * ROT_MAGNITUDE),
    ));

    // Generate A's camera points.
    let cams_a = (0..SAMPLE_POINTS)
        .map(|_| {
            let mut a = Vector3::new_random() * POINT_BOX_SIZE;
            a.x -= 0.5 * POINT_BOX_SIZE;
            a.y -= 0.5 * POINT_BOX_SIZE;
            a.z += 2.0;
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
