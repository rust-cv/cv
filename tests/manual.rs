use cv_core::nalgebra::{
    dimension::{U3, U8, U9},
    Isometry3, Matrix3, MatrixMN, UnitQuaternion, Vector2, Vector3, VectorN,
};
use cv_core::sample_consensus::Model;
use cv_core::{
    CameraPoint, EssentialMatrix, KeyPointsMatch, NormalizedKeyPoint, RelativeCameraPose,
};

const SAMPLE_POINTS: usize = 16;

const ROT_MAGNITUDE: f64 = 0.1;
const NEAR: f32 = 0.1;
const EPS_ROTATION: f64 = 1e-6;
const ITER_ROTATION: usize = 50;

const EIGHT_POINT_EIGEN_CONVERGENCE: f64 = 1e-6;
const EIGHT_POINT_EIGEN_ITERATIONS: usize = 50;

fn to_five(a: [NormalizedKeyPoint; SAMPLE_POINTS]) -> [NormalizedKeyPoint; 5] {
    [a[0], a[1], a[2], a[3], a[4]]
}

fn to_eight(a: [NormalizedKeyPoint; SAMPLE_POINTS]) -> [NormalizedKeyPoint; 8] {
    [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]]
}

#[test]
fn five_points_nullspace_basis() {
    let (_, _, kpa, kpb, _) = some_test_data();
    let e_basis = nister_stewenius::five_points_nullspace_basis(&to_five(kpa), &to_five(kpb))
        .expect("unable to compute nullspace basis");
    for s in 0..4 {
        let mut e = Matrix3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                e[(i, j)] = e_basis[(3 * i + j, s)];
            }
        }

        for i in 0..5 {
            let a = kpa[i].epipolar_point().0.coords;
            let b = kpb[i].epipolar_point().0.coords;

            let dot = b.dot(&(e * a)).abs();

            assert!(dot < NEAR as f64, "{} not small enough", dot);
        }
    }
}

#[test]
fn five_points_relative_pose() {
    let (real_pose, _, kpa, kpb, _) = some_test_data();

    eprintln!("\n8-pt:");
    let eight_essential = eight_point(&to_eight(kpa), &to_eight(kpb)).unwrap();
    eprintln!("{:?}", eight_essential);
    // Assert that the eight point essential works fine.
    for (&b, &a) in kpa.iter().zip(&kpb) {
        let residual = eight_essential.residual(&KeyPointsMatch(a, b));
        eprintln!("residual: {:?}", residual);
        assert!(residual.abs() < 0.1);
    }
    // Compute pose from essential and kp depths.
    let [rot_a, rot_b] = eight_essential
        .possible_rotations(EPS_ROTATION, ITER_ROTATION)
        .unwrap();
    // Convert rotations into quaternion form.
    let rot_from_real = |uquat| real_pose.rotation.rotation_to(&uquat).angle();
    let quat_a = UnitQuaternion::from(rot_a);
    let quat_b = UnitQuaternion::from(rot_b);
    eprintln!("rota: {:?}", rot_from_real(quat_a));
    eprintln!("rotb: {:?}", rot_from_real(quat_b));

    // Do the 5 point test.
    eprintln!("\n5-pt:");
    let mut essentials = nister_stewenius::five_points_relative_pose(&to_five(kpb), &to_five(kpa));

    let any_good = essentials.any(|essential| {
        for (&a, &b) in kpa.iter().zip(&kpb) {
            let residual = essential.residual(&KeyPointsMatch(b, a));
            eprintln!("residual: {:?}", residual);
            assert!(residual.abs() < 0.1);
        }

        eprintln!("essential: {:?}", essential);

        // Compute pose from essential and kp depths.
        let [rot_a, rot_b] = essential
            .possible_rotations(EPS_ROTATION, ITER_ROTATION)
            .unwrap();
        // Convert rotations into quaternion form.
        let quat_a = UnitQuaternion::from(rot_a);
        let quat_b = UnitQuaternion::from(rot_b);
        eprintln!("rota: {:?}", rot_from_real(quat_a));
        eprintln!("rotb: {:?}", rot_from_real(quat_b));

        // Extract vector from quaternion.
        let qcoord = |uquat: UnitQuaternion<f64>| uquat.quaternion().coords;
        // Compute residual via cosine distance of quaternions (guaranteed positive w).
        let a_close = 1.0 - qcoord(quat_a).dot(&qcoord(real_pose.rotation)) < 1e-6;
        let b_close = 1.0 - qcoord(quat_b).dot(&qcoord(real_pose.rotation)) < 1e-6;
        // At least one rotation is correct.
        a_close || b_close
    });

    assert!(any_good);
}

/// Gets a random relative pose, input points A, and input points B.
fn some_test_data() -> (
    RelativeCameraPose,
    EssentialMatrix,
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
    let cams_a = (0..SAMPLE_POINTS).map(|_| {
        let mut a = Vector3::new_random();
        a.x -= 0.5;
        a.y -= 0.5;
        a.z += 2.0;
        CameraPoint(a.into())
    });

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

fn encode_epipolar_equation_8(
    a: &[NormalizedKeyPoint; 8],
    b: &[NormalizedKeyPoint; 8],
) -> MatrixMN<f64, U8, U9> {
    let mut out: MatrixMN<f64, U8, U9> = nalgebra::zero();
    for i in 0..8 {
        let mut row = VectorN::<f64, U9>::zeros();
        let ap = a[i].epipolar_point().0.coords;
        let bp = b[i].epipolar_point().0.coords;
        for j in 0..3 {
            let v = bp[j] * ap;
            row.fixed_rows_mut::<U3>(3 * j).copy_from(&v);
        }
        out.row_mut(i).copy_from(&row.transpose());
    }
    out
}

pub fn eight_point(
    a: &[NormalizedKeyPoint; 8],
    b: &[NormalizedKeyPoint; 8],
) -> Option<EssentialMatrix> {
    let epipolar_constraint = encode_epipolar_equation_8(a, b);
    let eet = epipolar_constraint.transpose() * epipolar_constraint;
    let symmetric_eigens =
        eet.try_symmetric_eigen(EIGHT_POINT_EIGEN_CONVERGENCE, EIGHT_POINT_EIGEN_ITERATIONS)?;
    let eigenvector = symmetric_eigens
        .eigenvalues
        .iter()
        .enumerate()
        .min_by_key(|&(_, &n)| float_ord::FloatOrd(n))
        .map(|(ix, _)| symmetric_eigens.eigenvectors.column(ix).into_owned())?;
    Some(EssentialMatrix(Matrix3::from_iterator(
        eigenvector.iter().copied(),
    )))
}
