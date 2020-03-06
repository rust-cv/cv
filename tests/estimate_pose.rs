use arrsac::{Arrsac, Config as ArrsacConfig};
use cv_core::nalgebra::{Point2, Vector2};
use cv_core::sample_consensus::{Consensus, Model};
use cv_core::{CameraIntrinsics, ImageKeyPoint, KeyPointsMatch};
use log::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use space::{Bits512, Hamming};
use std::path::Path;

const LOWES_RATIO: f32 = 0.5;

type Descriptor = Hamming<Bits512>;

#[test]
fn estimate_pose() {
    env_logger::builder()
        .format_timestamp(Some(env_logger::TimestampPrecision::Millis))
        .init();
    // Intrinsics retrieved from calib_cam_to_cam.txt K_00.
    let intrinsics = CameraIntrinsics {
        focals: Vector2::new(9.842_439e2, 9.808_141e2),
        principal_point: Point2::new(6.9e2, 2.331_966e2),
        skew: 0.0,
    };

    info!("Extracting features");
    let (kps1, ds1) = image_to_kps("res/0000000000.png");
    let (kps2, ds2) = image_to_kps("res/0000000014.png");

    // This ensures the underlying algorithm does not change
    // by making sure that we get the exact expected number of features.
    assert_eq!(ds1.len(), 579);
    assert_eq!(ds2.len(), 500);

    info!(
        "Running matching on {} and {} descriptors",
        ds1.len(),
        ds2.len()
    );
    let matches: Vec<KeyPointsMatch> = match_descriptors(&ds1, &ds2)
        .into_iter()
        .map(|(ix1, ix2)| {
            let a = intrinsics.normalize(kps1[ix1]);
            let b = intrinsics.normalize(kps2[ix2]);
            KeyPointsMatch(a, b)
        })
        .collect();
    info!("Finished matching with {} matches", matches.len());

    // Run ARRSAC with the eight-point algorithm.
    info!("Running ARRSAC");
    let mut arrsac = Arrsac::new(ArrsacConfig::new(0.001), Pcg64::from_seed([1; 32]));
    let eight_point = eight_point::EightPoint::new();
    let (_, inliers) = arrsac
        .model_inliers(&eight_point, matches.iter().copied())
        .expect("failed to estimate model");
    info!("inliers: {}", inliers.len());

    // Ensures the underlying algorithms don't change at all.
    assert_eq!(inliers.len(), 35);
}

fn image_to_kps(path: impl AsRef<Path>) -> (Vec<ImageKeyPoint>, Vec<Descriptor>) {
    let mut akaze_config = akaze::Config::default();
    akaze_config.detector_threshold = 0.01;
    let (_, akaze_kps, akaze_ds) = akaze::extract_features(path, akaze_config);
    let kps = akaze_kps
        .into_iter()
        .map(|akp| ImageKeyPoint(Point2::new(akp.point.0 as f64, akp.point.1 as f64)))
        .collect();
    let ds = akaze_ds
        .into_iter()
        .map(|akaze::Descriptor { vector }| {
            let mut arr = [0; 512 / 8];
            arr[0..61].copy_from_slice(&vector);
            Hamming(Bits512(arr))
        })
        .collect();
    (kps, ds)
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
