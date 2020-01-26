use cv_core::nalgebra::{
    dimension::{U3, U5},
    Isometry3, Matrix3, MatrixMN, UnitQuaternion, Vector2, Vector3,
};
use cv_core::sample_consensus::Model;
use cv_core::{
    CameraPoint, EssentialMatrix, KeyPointsMatch, NormalizedKeyPoint, RelativeCameraPose,
};
use itertools::izip;

const NEAR: f32 = 1e-4;

#[test]
fn five_points_nullspace_basis() {
    let (_, _, kpa, kpb, _) = some_test_data();
    let e_basis = nister_stewenius::five_points_nullspace_basis(&kpa, &kpb)
        .expect("unable to compute nullspace basis");
    for s in 0..4 {
        let mut e = Matrix3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                e[(i, j)] = e_basis[(3 * i + j, s)];
            }
        }

        for i in 0..5 {
            let a = kpa[i].epipolar_point().0;
            let b = kpb[i].epipolar_point().0;

            let dot = b.dot(&(e * a)).abs();

            assert!(dot < NEAR, "{} not small enough", dot);
        }
    }
}

#[test]
fn five_points_relative_pose() {
    let (real_pose, real_essential, kpa, kpb, depths) = some_test_data();

    let essentials = nister_stewenius::five_points_relative_pose(&kpa, &kpb);

    for essential in essentials {
        eprintln!("essential guess: {:?}", essential);
        eprintln!("essential  real: {:?}", real_essential);
        for (&a, &b) in kpa.iter().zip(&kpb) {
            let residual = essential.residual(&KeyPointsMatch(a, b));
            eprintln!("residual: {:?}", residual);
            assert!(residual.abs() < 1e-5);
        }

        // Compute pose from essential and kp depths.
        let pose = essential.solve_pose(
            0.5,
            100,
            izip!(depths.clone(), kpa.iter().copied(), kpb.iter().copied()),
        );
        eprintln!("pose guess: {:?}", pose);
        eprintln!("pose  real: {:?}", real_pose);
    }
    panic!();
}

/// Gets a random relative pose, input points A, and input points B.
fn some_test_data() -> (
    RelativeCameraPose,
    EssentialMatrix,
    [NormalizedKeyPoint; 5],
    [NormalizedKeyPoint; 5],
    impl Iterator<Item = f32> + Clone,
) {
    // The relative pose orientation is fixed and translation is random.
    let relative_pose = RelativeCameraPose(Isometry3::from_parts(
        Vector3::new_random().into(),
        UnitQuaternion::from_euler_angles(0.1, 0.1, 0.1),
    ));

    // Generate A's camera points.
    let cams_a = (0..5).map(|_| {
        let mut a = Vector3::new_random();
        a.x -= 0.5;
        a.y -= 0.5;
        a.z += 2.0;
        CameraPoint(a.into())
    });

    // Generate B's camera points.
    let cams_b = cams_a.clone().map(|a| relative_pose.transform(a));

    let mut kps_a = [NormalizedKeyPoint(Vector2::zeros().into()); 5];
    for (keypoint, camera) in kps_a.iter_mut().zip(cams_a.clone()) {
        *keypoint = camera.into();
    }
    let mut kps_b = [NormalizedKeyPoint(Vector2::zeros().into()); 5];
    for (keypoint, camera) in kps_b.iter_mut().zip(cams_b.clone()) {
        *keypoint = camera.into();
    }

    (
        relative_pose,
        EssentialMatrix(
            *relative_pose.0.rotation.to_rotation_matrix().matrix()
                * relative_pose.0.translation.vector.cross_matrix(),
        ),
        kps_a,
        kps_b,
        cams_a.map(|p| p.0.z),
    )
}
