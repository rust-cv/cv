use akaze::Akaze;
use arrsac::Arrsac;
use bitarray::{BitArray, Hamming};
use cv_core::{
    nalgebra::{Point2, Vector2},
    sample_consensus::Consensus,
    CameraModel, FeatureMatch,
};
use log::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use space::Knn;
use std::path::Path;

const LOWES_RATIO: f32 = 0.5;

type Descriptor = BitArray<64>;
type Match = FeatureMatch;

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
    assert_eq!(ds1.len(), 399);
    assert_eq!(ds2.len(), 343);

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
    assert_eq!(matches.len(), 11);

    // Run ARRSAC with the eight-point algorithm.
    info!("Running ARRSAC");
    let mut arrsac = Arrsac::new(0.1, Pcg64::from_seed([1; 32]));
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
    assert_eq!(inliers.len(), 11);
}

fn match_descriptors(ds1: &[Descriptor], ds2: &[Descriptor]) -> Vec<(usize, usize)> {
    let two_neighbors = ds1
        .iter()
        .map(|d1| {
            let neighbors = space::LinearKnn {
                metric: Hamming,
                iter: ds2.iter(),
            }
            .knn(d1, 2);
            assert_eq!(neighbors.len(), 2, "there should be at least two matches");
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
