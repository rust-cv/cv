use akaze::types::evolution::Config as AkazeConfig;
use akaze::types::keypoint::Descriptor as AkazeDescriptor;
use arrsac::{Arrsac, Config as ArrsacConfig};
use cv_core::nalgebra::{Point2, Vector2};
use cv_core::sample_consensus::Consensus;
use cv_core::{CameraIntrinsics, ImageKeyPoint, KeyPointsMatch};
use eight_point::EightPoint;
use std::path::PathBuf;
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
    #[structopt(short, long, default_value = "500")]
    match_threshold: u32,
    /// Input folder with image files
    ///
    /// Kitti 2011_09_26
    #[structopt(parse(from_os_str))]
    input: PathBuf,
}

fn main() {
    let opt = Opt::from_args();
    // AKAZE extraction configuration
    let akaze_config = AkazeConfig::default();
    let images = opt
        .input
        .join("2011_09_26_drive_0035_extract")
        .join("image_00")
        .join("data");
    let paths = vec![images.join("0000000000.png"), images.join("0000000001.png")];
    let mut image_points = paths
        .into_iter()
        .map(|path| akaze::extract_features(path, akaze_config))
        .map(|(_, kps, descriptors)| {
            (
                kps.into_iter()
                    .map(|kp| ImageKeyPoint(Point2::new(kp.point.0 as f64, kp.point.1 as f64)))
                    .collect::<Vec<_>>(),
                descriptors
                    .into_iter()
                    .map(|AkazeDescriptor { vector }| vector)
                    .collect::<Vec<_>>(),
            )
        });
    // Intrinsics retrieved from calib_cam_to_cam.txt K_00.
    let intrinsics = CameraIntrinsics {
        focals: Vector2::new(9.842_439e2, 9.808_141e2),
        principal_point: Point2::new(6.9e2, 2.331_966e2),
        skew: 0.0,
    };
    let (a_kps, a_descriptors) = image_points.next().unwrap();
    let (b_kps, b_descriptors) = image_points.next().unwrap();
    let forward_matches = matching(&a_descriptors, &b_descriptors);
    let reverse_matches = matching(&b_descriptors, &a_descriptors);
    let matches = forward_matches
        .iter()
        .enumerate()
        .filter_map(|(aix, &(bix, distance))| {
            let is_symmetric = reverse_matches[bix].0 == aix;
            let in_threshold = distance < opt.match_threshold;
            if is_symmetric && in_threshold {
                let a = intrinsics.normalize(a_kps[aix]);
                let b = intrinsics.normalize(b_kps[bix]);
                Some(KeyPointsMatch(a, b))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    let eight_point = EightPoint::new();
    let mut arrsac = Arrsac::new(ArrsacConfig::new(0.01), rand::thread_rng());
    let essential = arrsac.model(&eight_point, matches.iter().copied()).unwrap();
    let pose = essential
        .solve_unscaled_pose(
            1e-6,
            100,
            0.5,
            cv_core::geom::make_one_pose_dlt_triangulator(1e-6, 100),
            matches.iter().copied(),
        )
        .unwrap();
    eprintln!("pose: {:?}", pose);
    eprintln!("rotation: {:?}", pose.rotation.angle());
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
