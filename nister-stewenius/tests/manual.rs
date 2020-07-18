use cv_core::nalgebra::{
    dimension::{U3, U8, U9},
    Isometry3, Matrix3, MatrixMN, MatrixN, UnitQuaternion, Vector2, Vector3, VectorN,
};
use cv_core::sample_consensus::Model;
use cv_core::{
    CameraPoint, EssentialMatrix, KeyPointsMatch, NormalizedKeyPoint, RelativeCameraPose,
};

const SAMPLE_POINTS: usize = 16;

const ROT_MAGNITUDE: f64 = 0.3;
const TRANS_MAGNITUDE: f64 = 2.0;
const NEAR: f32 = 0.1;
const EPS_ROTATION: f64 = 1e-9;
const ITER_ROTATION: usize = 500;

const EIGHT_POINT_EIGEN_CONVERGENCE: f64 = 1e-12;
const EIGHT_POINT_EIGEN_ITERATIONS: usize = 500;

fn to_five(a: [NormalizedKeyPoint; SAMPLE_POINTS]) -> [NormalizedKeyPoint; 5] {
    [a[0], a[1], a[2], a[3], a[4]]
}

fn to_eight(a: [NormalizedKeyPoint; SAMPLE_POINTS]) -> [NormalizedKeyPoint; 8] {
    [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]]
}

#[test]
fn five_points_nullspace_basis() {
    let (_, _, kpa, kpb, kpr, _) = some_test_data();
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
    let (real_pose, _, kpa, kpb, kpr, _) = some_test_data();

    eprintln!("\n8-pt:");
    let eight_essential = eight_point(&to_eight(kpa), &to_eight(kpb)).unwrap();
    assert_essential(eight_essential);
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
    // Assert that the eight point essential works fine.
    for ((&b, &a), &r) in kpa.iter().zip(&kpb).zip(&kpr) {
        let residual_match = eight_essential.residual(&KeyPointsMatch(b, a));
        eprintln!("residual match: {:?}", residual_match.abs());
        let residual_wrong = eight_essential.residual(&KeyPointsMatch(a, r));
        eprintln!("residual wrong: {:?}", residual_wrong.abs());
        assert!(residual_match.abs() < 0.1);
    }

    // Do the 5 point test.
    eprintln!("\n5-pt:");
    let essentials = nister_stewenius::five_points_relative_pose(&to_five(kpb), &to_five(kpa))
        .collect::<Vec<_>>();

    let any_good = essentials.iter().any(|essential| {
        for ((&b, &a), &r) in kpa.iter().zip(&kpb).zip(&kpr) {
            let residual = essential.residual(&KeyPointsMatch(a, b));
            eprintln!("residual: {:?}", residual.abs());
            let residual_wrong = eight_essential.residual(&KeyPointsMatch(a, r));
            eprintln!("residual wrong: {:?}", residual_wrong.abs());
            assert!(residual.abs() < 1.0);
        }

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

/// Gets a random relative pose, input points A, input points B, and random points.
fn some_test_data() -> (
    RelativeCameraPose,
    EssentialMatrix,
    [NormalizedKeyPoint; SAMPLE_POINTS],
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
            let mut a = Vector3::new_random() * TRANS_MAGNITUDE;
            a.x -= 0.5 * TRANS_MAGNITUDE;
            a.y -= 0.5 * TRANS_MAGNITUDE;
            a.z += 2.0;
            CameraPoint(a.into())
        })
        .collect::<Vec<_>>()
        .into_iter();

    let cams_rand = (0..SAMPLE_POINTS).map(|_| {
        let mut a = Vector3::new_random() * TRANS_MAGNITUDE;
        a.x -= 0.5 * TRANS_MAGNITUDE;
        a.y -= 0.5 * TRANS_MAGNITUDE;
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
    let mut kps_rand = [NormalizedKeyPoint(Vector2::zeros().into()); SAMPLE_POINTS];
    for (keypoint, camera) in kps_rand.iter_mut().zip(cams_rand) {
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
        kps_rand,
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
            let v = ap[j] * bp;
            row.fixed_rows_mut::<U3>(3 * j).copy_from(&v);
        }
        out.row_mut(i).copy_from(&row.transpose());
    }
    out
}

pub fn recondition_matrix(mat: Matrix3<f64>) -> EssentialMatrix {
    let old_svd = mat.svd(true, true);
    // We need to sort the singular values in the SVD.
    let mut sources = [0, 1, 2];
    sources.sort_unstable_by_key(|&ix| float_ord::FloatOrd(-old_svd.singular_values[ix]));
    let mut svd = old_svd;
    for (dest, &source) in sources.iter().enumerate() {
        svd.singular_values[dest] = old_svd.singular_values[source];
        svd.u
            .as_mut()
            .unwrap()
            .column_mut(dest)
            .copy_from(&old_svd.u.as_ref().unwrap().column(source));
        svd.v_t
            .as_mut()
            .unwrap()
            .row_mut(dest)
            .copy_from(&old_svd.v_t.as_ref().unwrap().row(source));
    }
    // Now that the singular values are sorted, find the closest
    // essential matrix to E in frobenius form.
    // This consists of averaging the two non-zero singular values
    // and zeroing out the near-zero singular value.
    svd.singular_values[2] = 0.0;
    let new_singular = (svd.singular_values[0] + svd.singular_values[1]) / 2.0;
    svd.singular_values[0] = new_singular;
    svd.singular_values[1] = new_singular;

    let mat = svd.recompose().unwrap();
    EssentialMatrix(mat)
}

pub fn eight_point(
    a: &[NormalizedKeyPoint; 8],
    b: &[NormalizedKeyPoint; 8],
) -> Option<EssentialMatrix> {
    let epipolar_constraint = encode_epipolar_equation_8(a, b);
    let eet = epipolar_constraint.transpose() * epipolar_constraint;
    let eigens =
        eet.try_symmetric_eigen(EIGHT_POINT_EIGEN_CONVERGENCE, EIGHT_POINT_EIGEN_ITERATIONS)?;
    let eigenvector = eigens
        .eigenvalues
        .iter()
        .enumerate()
        .min_by_key(|&(_, &n)| float_ord::FloatOrd(n))
        .map(|(ix, _)| eigens.eigenvectors.column(ix).into_owned())?;
    let mat = Matrix3::from_iterator(eigenvector.iter().copied());
    Some(recondition_matrix(mat))
}

fn assert_essential(EssentialMatrix(e): EssentialMatrix) {
    // TODO: Figure out how to fix this.
    assert!(
        e.determinant() < 0.1,
        "matrix determinant not near zero: {}",
        e.determinant()
    );
    let o = 2.0 * e * e.transpose() * e - (e * e.transpose()).trace() * e;
    for &n in o.iter() {
        assert!(n < 0.1, "matrix not near zero: {:?}", o);
    }
}
