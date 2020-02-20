use akaze::types::evolution::Config as AkazeConfig;
use akaze::types::keypoint::Descriptor as AkazeDescriptor;
use arrsac::{Arrsac, Config as ArrsacConfig};
use cv_core::nalgebra::{Point2, Vector2};
use cv_core::sample_consensus::{Consensus, Model};
use cv_core::{CameraIntrinsics, ImageKeyPoint, KeyPointsMatch};
use eight_point::EightPoint;
use log::info;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::path::PathBuf;
use std::sync::mpsc::{sync_channel, Receiver};
use structopt::StructOpt;

#[derive(StructOpt)]
#[structopt(
    name = "vslam-sandbox",
    about = "A tool for testing vslam algorithms.",
    rename_all = "kebab-case"
)]
struct Opt {
    /// The threshold in bits for matching.
    ///
    /// Setting this to a high number disables it.
    #[structopt(short, long, default_value = "64")]
    match_threshold: u32,
    /// The threshold for ARRSAC.
    #[structopt(short, long, default_value = "0.001")]
    arrsac_threshold: f32,
    /// List of image files
    ///
    /// Must be Kitti 2011_09_26 camera 0
    #[structopt(parse(from_os_str))]
    images: Vec<PathBuf>,
}

fn main() {
    env_logger::builder()
        .format_timestamp(Some(env_logger::TimestampPrecision::Millis))
        .init();
    let opt = Opt::from_args();
    // Intrinsics retrieved from calib_cam_to_cam.txt K_00.
    let intrinsics = CameraIntrinsics {
        focals: Vector2::new(9.842_439e2, 9.808_141e2),
        principal_point: Point2::new(6.9e2, 2.331_966e2),
        skew: 0.0,
    };

    // Create a channel that will produce features in another parallel thread.
    let features = features_stream(opt.images.clone());
    let mut prev = features.recv().unwrap();
    for next in features {
        info!("prev kps: {}", prev.0.len());
        info!("next kps: {}", next.0.len());
        let forward_matches = matching(&prev.1, &next.1);
        let reverse_matches = matching(&next.1, &prev.1);
        let matches = forward_matches
            .iter()
            .enumerate()
            .filter_map(|(aix, &(bix, distance))| {
                let is_symmetric = reverse_matches[bix].0 == aix;
                let in_threshold = distance < opt.match_threshold;
                if is_symmetric && in_threshold {
                    let a = intrinsics.normalize(prev.0[aix]);
                    let b = intrinsics.normalize(next.0[bix]);
                    Some(KeyPointsMatch(a, b))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        info!("matches: {}", matches.len());
        let eight_point = EightPoint::new();
        let mut arrsac = Arrsac::new(
            ArrsacConfig::new(opt.arrsac_threshold),
            Pcg64::from_seed([1; 32]),
        );
        let (essential, inliers) = arrsac
            .model_inliers(&eight_point, matches.iter().copied())
            .unwrap();
        info!("inliers: {}", inliers.len());
        let residual_average = inliers
            .iter()
            .map(|&ix| essential.residual(&matches[ix]).abs())
            .sum::<f32>()
            / inliers.len() as f32;
        info!("inlier residual average: {}", residual_average);
        let pose = essential
            .solve_unscaled_pose(
                1e-6,
                100,
                0.5,
                cv_core::geom::make_one_pose_dlt_triangulator(1e-6, 100),
                matches.iter().copied(),
            )
            .unwrap();
        info!("rotation: {:?}", pose.rotation.angle());
        prev = next;
    }
}

fn features_stream(paths: Vec<PathBuf>) -> Receiver<(Vec<ImageKeyPoint>, Vec<Vec<u8>>)> {
    let (tx, rx) = sync_channel(5);

    std::thread::spawn(move || {
        for path in paths {
            tx.send(kps_descriptors(path)).unwrap();
        }
    });

    rx
}

fn kps_descriptors(path: PathBuf) -> (Vec<ImageKeyPoint>, Vec<Vec<u8>>) {
    let akaze_config = AkazeConfig::default();
    let (_, kps, descriptors) = akaze::extract_features(path, akaze_config);
    (
        kps.into_iter()
            .map(|kp| ImageKeyPoint(Point2::new(kp.point.0 as f64, kp.point.1 as f64)))
            .collect::<Vec<_>>(),
        descriptors
            .into_iter()
            .map(|AkazeDescriptor { vector }| vector)
            .collect::<Vec<_>>(),
    )
}

fn distance(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).map(|(&a, &b)| (a ^ b).count_ones()).sum()
}

fn matching(a_descriptors: &[Vec<u8>], b_descriptors: &[Vec<u8>]) -> Vec<(usize, u32)> {
    a_descriptors
        .iter()
        .map(|a| {
            b_descriptors
                .iter()
                .map(|b| distance(a, b))
                .enumerate()
                .min_by_key(|&(_, d)| d)
                .unwrap()
        })
        .collect::<Vec<_>>()
}
