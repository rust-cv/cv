use cv_core::nalgebra::{
    dimension::{U3, U5},
    Isometry3, Matrix3, MatrixMN, UnitQuaternion, Vector3,
};
use cv_core::sample_consensus::Model;
use cv_core::{
    CameraPoint, EssentialMatrix, KeyPointsMatch, NormalizedKeyPoint, RelativeCameraPose,
};
use itertools::izip;

const NEAR: f32 = 1e-4;

#[test]
fn five_points_nullspace_basis() {
    let (_, _, x1, x2) = some_test_data();
    let e_basis = nister_stewenius::five_points_nullspace_basis(&x1, &x2)
        .expect("unable to compute nullspace basis");
    for s in 0..4 {
        let mut e = Matrix3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                e[(i, j)] = e_basis[(3 * i + j, s)];
            }
        }

        for i in 0..5 {
            let a = x1.column(i).into_owned();
            let b = x2.column(i).into_owned();

            let dot = b.dot(&(e * a)).abs();

            assert!(dot < NEAR, "{} not small enough", dot);
        }
    }
}

#[test]
fn five_points_relative_pose() {
    let (_, real_essential, x1, x2) = some_test_data();

    let essentials = nister_stewenius::five_points_relative_pose(&x1, &x2);

    for essential in essentials {
        eprintln!("essential guess: {:?}", essential);
        eprintln!("essential  real: {:?}", real_essential);
        for (a, b) in x1.column_iter().zip(x2.column_iter()) {
            let residual = (b.transpose() * essential.0 * a)[0];
            eprintln!("residual: {:?}", residual);
            assert!(residual.abs() < 1e-5);
        }
    }
}

/// Gets a random relative pose, input points A, and input points B.
fn some_test_data() -> (
    RelativeCameraPose,
    EssentialMatrix,
    MatrixMN<f32, U3, U5>,
    MatrixMN<f32, U3, U5>,
) {
    let mut x = MatrixMN::<f32, U3, U5>::new_random();
    for v in x.rows_mut(0, 2).iter_mut() {
        *v -= 0.5;
    }
    for v in x.row_mut(2).iter_mut() {
        *v += 2.0;
    }
    let relative_pose = RelativeCameraPose(Isometry3::from_parts(
        Vector3::new_random().into(),
        UnitQuaternion::from_euler_angles(0.1, 0.1, 0.1),
    ));
    let mut y = x.clone();
    for mut col in y.column_iter_mut() {
        let ccol = col.clone_owned();
        col.copy_from(&relative_pose.transform(CameraPoint(ccol)).0);
    }
    // Normalize all the keypoints to hide the depth information (just bearings).
    for mut col in x.column_iter_mut().chain(y.column_iter_mut()) {
        col.x /= col.z;
        col.y /= col.z;
        col.z = 1.0;
    }
    (
        relative_pose,
        EssentialMatrix(
            *relative_pose.0.rotation.to_rotation_matrix().matrix()
                * relative_pose.0.translation.vector.cross_matrix(),
        ),
        x,
        y,
    )
}
