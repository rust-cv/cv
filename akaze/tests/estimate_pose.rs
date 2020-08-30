use akaze::Akaze;
use arrsac::Arrsac;
use bitarray::BitArray;
use cv_core::nalgebra::{Point2, Vector2};
use cv_core::sample_consensus::Consensus;
use cv_core::{CameraModel, FeatureMatch};
use log::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::path::Path;

const LOWES_RATIO: f32 = 0.5;

type Descriptor = BitArray<64>;
type Match = FeatureMatch<cv_pinhole::NormalizedKeyPoint>;

fn image_to_kps(path: impl AsRef<Path>) -> (Vec<akaze::KeyPoint>, Vec<Descriptor>) {
    Akaze::sparse().extract_path(path).unwrap()
}

#[test]
fn estimate_pose() {
    pretty_env_logger::init_timed();
    // Intrinsics retrieved from calib_cam_to_cam.txt K_00.
    let intrinsics = cv_pinhole::CameraIntrinsics {
        focals: Vector2::new(9.842_439e2, 9.808_141e2),
        principal_point: Point2::new(6.9e2, 2.331_966e2),
        skew: 0.0,
    };

    // Extract features with AKAZE.
    info!("Extracting features");
    let (kps1, ds1) = image_to_kps("../res/0000000000.png");
    let (kps2, ds2) = image_to_kps("../res/0000000014.png");

    // This ensures the underlying algorithm does not change
    // by making sure that we get the exact expected number of features.
    assert_eq!(ds1.len(), 575);
    assert_eq!(ds2.len(), 497);

    // Perform matching.
    info!(
        "Running matching on {} and {} descriptors",
        ds1.len(),
        ds2.len()
    );
    let matches: Vec<Match> = match_descriptors(&ds1, &ds2)
        .into_iter()
        .map(|(ix1, ix2)| {
            let a = intrinsics.calibrate(kps1[ix1]);
            let b = intrinsics.calibrate(kps2[ix2]);
            FeatureMatch(a, b)
        })
        .collect();
    info!("Finished matching with {} matches", matches.len());
    assert_eq!(matches.len(), 35);

    // Run ARRSAC with the eight-point algorithm.
    info!("Running ARRSAC");
    let mut arrsac = Arrsac::new(0.001, Pcg64::from_seed([1; 32]));
    let eight_point = eight_point::EightPoint::new();
    let (_, inliers) = arrsac
        .model_inliers(&eight_point, matches.iter().copied())
        .expect("failed to estimate model");
    info!("inliers: {}", inliers.len());
    info!(
        "inlier ratio: {}",
        inliers.len() as f32 / matches.len() as f32
    );

    // Ensures the underlying algorithms don't change at all.
    assert_eq!(inliers.len(), 35);
}

fn match_descriptors(ds1: &[Descriptor], ds2: &[Descriptor]) -> Vec<(usize, usize)> {
    use space::Neighbor;
    let two_neighbors = ds1
        .iter()
        .map(|d1| {
            let mut neighbors = [Neighbor::invalid(); 2];
            assert_eq!(
                space::linear_knn(d1, &mut neighbors, ds2).len(),
                2,
                "there should be at least two matches"
            );
            neighbors
        })
        .enumerate();
    let satisfies_lowes_ratio = two_neighbors.filter(|(_, neighbors)| {
        (neighbors[0].distance as f32) < neighbors[1].distance as f32 * LOWES_RATIO
    });
    satisfies_lowes_ratio
        .map(|(ix1, neighbors)| (ix1, neighbors[0].index))
        .collect()
}
