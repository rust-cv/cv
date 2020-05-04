use approx::assert_relative_eq;
use arraymap::ArrayMap;
use arrsac::{Arrsac, Config};
use cv_core::nalgebra::{IsometryMatrix3, Point3, Translation, Rotation3, Vector3};
use lambda_twist::LambdaTwist;
use rand::{rngs::SmallRng, SeedableRng};
use cv_core::sample_consensus::Consensus;
use cv_core::FeatureWorldMatch;
use cv_pinhole::NormalizedKeyPoint;

const EPSILON_APPROX: f64 = 1e-6;

#[test]
fn arrsac_manual() {
    let mut arrsac = Arrsac::new(Config::new(0.01), SmallRng::from_seed([0; 16]));

    // Define some points in camera coordinates (with z > 0).
    let camera_depth_points = [
        [-0.228_125, -0.061_458_334, 1.0],
        [0.418_75, -0.581_25, 2.0],
        [1.128_125, 0.878_125, 3.0],
        [-0.528_125, 0.178_125, 2.5],
        [-0.923_424, -0.235_125, 2.8],
    ]
    .map(|&p| Point3::from(p));

    // Define the camera pose.
    let rot = Rotation3::from_euler_angles(0.1, 0.2, 0.3);
    let trans = Translation::from(Vector3::new(0.1, 0.2, 0.3));
    let pose = IsometryMatrix3::from_parts(trans, rot);

    // Compute world coordinates.
    let world_points = camera_depth_points.map(|p| pose.inverse() * p);

    // Compute normalized image coordinates.
    let normalized_image_coordinates = camera_depth_points.map(|p| (p / p.z).xy());

    let samples: Vec<FeatureWorldMatch<NormalizedKeyPoint>> = world_points
        .iter()
        .zip(&normalized_image_coordinates)
        .map(|(&world, &image)| FeatureWorldMatch(image.into(), world.into()))
        .collect();

    // Estimate potential poses with P3P.
    // Arrsac should use the fourth point to filter and find only one model from the 4 generated.
    let pose = arrsac
        .model(&LambdaTwist::new(), samples.iter().cloned())
        .unwrap();

    // Compare the pose to ground truth.
    assert_relative_eq!(rot, pose.rotation, epsilon = EPSILON_APPROX);
    assert_relative_eq!(trans, pose.translation, epsilon = EPSILON_APPROX);
}
