use cv_core::nalgebra::{
    dimension::{U3, U5},
    Isometry3, MatrixMN, UnitQuaternion, Vector2, Vector3,
};
use cv_core::sample_consensus::Model;
use cv_core::{CameraPoint, KeyPointsMatch, NormalizedKeyPoint, RelativeCameraPose};
use itertools::izip;

#[test]
fn simple_non_degenerate_case() {
    let camera_points_a = [
        CameraPoint(Vector3::new(0.1, 0.4, 2.0)),
        CameraPoint(Vector3::new(-0.4, -0.66, 3.0)),
        CameraPoint(Vector3::new(-0.5, -0.16, 1.5)),
        CameraPoint(Vector3::new(0.25, -0.86, 4.5)),
        CameraPoint(Vector3::new(-0.77, 0.42, 5.0)),
        CameraPoint(Vector3::new(-1.1, -0.48, 2.2)),
        CameraPoint(Vector3::new(-0.421, 0.14, 4.134)),
        CameraPoint(Vector3::new(-0.48713, -0.920, 4.897)),
        CameraPoint(Vector3::new(0.1842, -0.92734, 1.3123)),
    ];
    let relative_pose = RelativeCameraPose(Isometry3::from_parts(
        Vector3::new(1.0, 2.0, 3.0).into(),
        UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3),
    ));
    let camera_points_b: Vec<CameraPoint> = camera_points_a
        .iter()
        .cloned()
        .map(|p| relative_pose.transform(p))
        .collect();

    let norm_image_coords_a: Vec<NormalizedKeyPoint> =
        camera_points_a.iter().cloned().map(|p| p.into()).collect();

    let norm_image_coords_b: Vec<NormalizedKeyPoint> =
        camera_points_b.iter().cloned().map(|p| p.into()).collect();

    let x1 = MatrixMN::<f32, U3, U5>::from_iterator(
        norm_image_coords_a.iter().flat_map(|p| p.iter().copied()),
    );

    let x2 = MatrixMN::<f32, U3, U5>::from_iterator(
        norm_image_coords_b.iter().flat_map(|p| p.iter().copied()),
    );

    let essentials = nister_stewenius::five_points_relative_pose(&x1, &x2);

    for essential in essentials {
        eprintln!("essential: {:?}", essential);
        for (&a, &b) in norm_image_coords_a.iter().zip(norm_image_coords_b.iter()) {
            let residual = essential.residual(&KeyPointsMatch(a, b));
            eprintln!("residual: {:?}", residual);
            // assert!(residual.abs() < 1e-5);
        }
        let depths = camera_points_a.iter().map(|p| p.z);
        let b_coords = norm_image_coords_b.iter().copied();
        let a_coords = norm_image_coords_a.iter().copied();
        let pose = essential.solve_pose(0.1, 500, izip!(depths, a_coords, b_coords));
        eprintln!("pose: {:?}", pose);
        eprintln!("actual: {:?}", relative_pose);
    }
    panic!();
}
