use cv_core::nalgebra::{Isometry3, UnitQuaternion, Vector2, Vector3};
use cv_core::{CameraPoint, NormalizedKeyPoint, RelativeCameraPose};

#[test]
fn simple_non_degenerate_case() {
    let camera_points_a = [
        CameraPoint(Vector3::new(0.1, 0.4, 2.0)),
        CameraPoint(Vector3::new(-0.4, -0.66, 3.0)),
        CameraPoint(Vector3::new(-0.5, -0.16, 1.5)),
        CameraPoint(Vector3::new(0.25, -0.86, 4.5)),
        CameraPoint(Vector3::new(-0.77, 0.42, 5.0)),
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

    let norm_image_coords_a: Vec<Vector2<f32>> = camera_points_a
        .iter()
        .cloned()
        .map(|p| p.into())
        .map(|p: NormalizedKeyPoint| p.0.coords)
        .collect();

    let norm_image_coords_b: Vec<Vector2<f32>> = camera_points_b
        .iter()
        .cloned()
        .map(|p| p.into())
        .map(|p: NormalizedKeyPoint| p.0.coords)
        .collect();

    let essentials = nister_stewenius::five_point_relative_pose(
        100,
        &norm_image_coords_a[..],
        &norm_image_coords_b[..],
    );

    for essential in essentials {
        for (a, b) in norm_image_coords_a.iter().zip(norm_image_coords_b.iter()) {
            let residual = (b.push(1.0).transpose() * essential * a.push(1.0))[0];
            eprintln!(
                "residual: {:?}",
                (b.push(1.0).transpose() * essential * a.push(1.0))[0]
            );
            assert!(residual.abs() < 1e-5);
        }
    }
}
