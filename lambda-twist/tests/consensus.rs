use approx::assert_relative_eq;
use arrayvec::ArrayVec;
use arrsac::Arrsac;
use cv_core::{
    nalgebra::{IsometryMatrix3, Point2, Point3, Rotation3, Translation, UnitVector3, Vector3},
    sample_consensus::Consensus,
    FeatureWorldMatch, Projective,
};
use lambda_twist::LambdaTwist;
use rand::{rngs::SmallRng, SeedableRng};

const EPSILON_APPROX: f64 = 1e-6;

fn map<T, U, F: Fn(T) -> U, const N: usize>(f: F, array: ArrayVec<T, N>) -> ArrayVec<U, N> {
    array.into_iter().map(f).collect()
}

#[test]
fn arrsac_manual() {
    let mut arrsac = Arrsac::new(0.01, SmallRng::seed_from_u64(0));

    // Define some points in camera coordinates (with z > 0).
    let camera_depth_points: ArrayVec<Point3<f64>, 5> = map(
        Point3::from,
        [
            [-0.228_125, -0.061_458_334, 1.0],
            [0.418_75, -0.581_25, 2.0],
            [1.128_125, 0.878_125, 3.0],
            [-0.528_125, 0.178_125, 2.5],
            [-0.923_424, -0.235_125, 2.8],
        ]
        .into(),
    );

    // Define the camera pose.
    let rot = Rotation3::from_euler_angles(0.1, 0.2, 0.3);
    let trans = Translation::from(Vector3::new(0.1, 0.2, 0.3));
    let pose = IsometryMatrix3::from_parts(trans, rot);

    // Compute world coordinates.
    let world_points = map(|p| pose.inverse() * p, camera_depth_points.clone());

    // Compute normalized image coordinates.
    let normalized_image_coordinates = map(|p| (p / p.z).xy(), camera_depth_points);

    let samples: Vec<FeatureWorldMatch> = world_points
        .iter()
        .zip(&normalized_image_coordinates)
        .map(|(&world, &image)| {
            FeatureWorldMatch(
                UnitVector3::new_normalize(image.to_homogeneous()),
                Projective::from_homogeneous(world.to_homogeneous()),
            )
        })
        .collect();

    // Estimate potential poses with P3P.
    // Arrsac should use the fourth point to filter and find only one model from the 4 generated.
    let pose = arrsac
        .model(&LambdaTwist::new(), samples.iter().cloned())
        .unwrap();

    // Compare the pose to ground truth.
    assert_relative_eq!(rot, pose.0.rotation, epsilon = EPSILON_APPROX);
    assert_relative_eq!(trans, pose.0.translation, epsilon = EPSILON_APPROX);
}

#[test]
fn endless_loop_case() {
    let mut arrsac = Arrsac::new(0.01, SmallRng::seed_from_u64(0));

    let samples = [
        FeatureWorldMatch(
            UnitVector3::new_normalize(
                Point2::new(0.3070512144698557, 0.19317668016026052).to_homogeneous(),
            ),
            Projective::from_point(Point3::new(1.0, 1.0, 0.0)),
        ),
        FeatureWorldMatch(
            UnitVector3::new_normalize(
                Point2::new(0.3208462966353674, 0.20741702947913013).to_homogeneous(),
            ),
            Projective::from_point(Point3::new(1.0, 1.5, 0.0)),
        ),
        FeatureWorldMatch(
            UnitVector3::new_normalize(
                Point2::new(0.3070512144698557, 0.19317668016026052).to_homogeneous(),
            ),
            Projective::from_point(Point3::new(3.0, 1.0, 0.0)),
        ),
        FeatureWorldMatch(
            UnitVector3::new_normalize(
                Point2::new(0.3208462966353674, 0.20741702947913013).to_homogeneous(),
            ),
            Projective::from_point(Point3::new(1.0, 2.0, 0.0)),
        ),
        FeatureWorldMatch(
            UnitVector3::new_normalize(
                Point2::new(0.3208462966353674, 0.20741702947913013).to_homogeneous(),
            ),
            Projective::from_point(Point3::new(2.0, 2.0, 0.0)),
        ),
        FeatureWorldMatch(
            UnitVector3::new_normalize(
                Point2::new(0.3070512144698557, 0.19317668016026052).to_homogeneous(),
            ),
            Projective::from_point(Point3::new(3.0, 2.0, 0.0)),
        ),
        FeatureWorldMatch(
            UnitVector3::new_normalize(
                Point2::new(0.26619553978146293, 0.15033756455213498).to_homogeneous(),
            ),
            Projective::from_point(Point3::new(1.0, 3.0, 0.0)),
        ),
        FeatureWorldMatch(
            UnitVector3::new_normalize(
                Point2::new(0.3494806979265859, 0.18264329458710366).to_homogeneous(),
            ),
            Projective::from_point(Point3::new(2.0, 3.0, 0.0)),
        ),
        FeatureWorldMatch(
            UnitVector3::new_normalize(
                Point2::new(0.32132193890323213, 0.15408143785084824).to_homogeneous(),
            ),
            Projective::from_point(Point3::new(3.0, 3.0, 0.0)),
        ),
    ];

    // Estimate potential poses with P3P.
    // Arrsac should use the fourth point to filter and find only one model from the 4 generated.
    arrsac
        .model(&LambdaTwist::new(), samples.iter().cloned())
        .unwrap();
}
