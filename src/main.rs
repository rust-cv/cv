mod export;
mod slam;

use cv::{
    camera::pinhole::{self, CameraIntrinsics, CameraIntrinsicsK1Distortion},
    consensus::Arrsac,
    estimate::EightPoint,
    feature::akaze,
    geom::MinimalSquareReprojectionErrorTriangulator,
    BitArray, CameraModel, Consensus, FeatureMatch, Model, TriangulatorRelative,
};

use cv::nalgebra::{Point2, Point3, Vector2};

use log::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver};
use structopt::StructOpt;

type Descriptor = BitArray<64>;

#[derive(StructOpt, Clone)]
#[structopt(name = "vslam-sandbox", about = "A tool for testing vslam algorithms.")]
struct Opt {
    /// The threshold in bits for matching.
    ///
    /// Setting this to a high number disables it.
    #[structopt(short, long, default_value = "64")]
    match_threshold: usize,
    /// The threshold for ARRSAC.
    #[structopt(short, long, default_value = "0.001")]
    arrsac_threshold: f32,
    /// The threshold for AKAZE.
    #[structopt(short = "z", long, default_value = "0.001")]
    akaze_threshold: f64,
    /// The threshold for reprojection error in pixels.
    #[structopt(long, default_value = "2.5")]
    reprojection_threshold: f64,
    /// The x focal length
    #[structopt(long, default_value = "984.2439")]
    x_focal: f64,
    /// The y focal length
    #[structopt(long, default_value = "980.8141")]
    y_focal: f64,
    /// The x optical center coordinate
    #[structopt(long, default_value = "690.0")]
    x_center: f64,
    /// The y optical center coordinate
    #[structopt(long, default_value = "233.1966")]
    y_center: f64,
    /// The skew
    #[structopt(long, default_value = "0.0")]
    skew: f64,
    /// The K1 radial distortion
    #[structopt(long, default_value = "0.0")]
    radial_distortion: f64,
    /// Output PLY file to deposit point cloud
    #[structopt(short, long)]
    output: Option<PathBuf>,
    /// List of image files
    ///
    /// Default vales are for Kitti 2011_09_26 camera 0
    #[structopt(parse(from_os_str))]
    images: Vec<PathBuf>,
}

fn main() {
    pretty_env_logger::init_timed();
    let opt = Opt::from_args();
    // Intrinsics retrieved from calib_cam_to_cam.txt K_00.
    let intrinsics = CameraIntrinsicsK1Distortion::new(
        CameraIntrinsics {
            focals: Vector2::new(opt.x_focal, opt.y_focal),
            principal_point: Point2::new(opt.x_center, opt.y_center),
            skew: opt.skew,
        },
        opt.radial_distortion,
    );

    let net_focal = (opt.x_focal.powi(2) + opt.y_focal.powi(2)).sqrt();

    // Create a channel that will produce features in another parallel thread.
    let features = features_stream(&opt);
    let mut prev = features.recv().unwrap();
    for next in features {
        info!("start frame");
        info!("prev kps: {}", prev.0.len());
        info!("next kps: {}", next.0.len());

        // Compute best matches from previous frame to next frame and vice versa.
        let forward_matches = matching(&prev.1, &next.1);
        let reverse_matches = matching(&next.1, &prev.1);

        // Compute the symmetric matches (matches that were the same going forwards and backwards).
        let matches = forward_matches
            .iter()
            .enumerate()
            .filter_map(|(aix, &(bix, distance))| {
                let is_symmetric = reverse_matches[bix].0 == aix;
                let in_threshold = distance < opt.match_threshold;
                if is_symmetric && in_threshold {
                    let a = intrinsics.calibrate(prev.0[aix]);
                    let b = intrinsics.calibrate(next.0[bix]);
                    Some(FeatureMatch(a, b))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        info!("matches: {}", matches.len());

        // Perform sample consensus.
        let eight_point = EightPoint::new();
        let mut arrsac = Arrsac::new(opt.arrsac_threshold, Pcg64::from_seed([1; 32]));
        let (essential, inliers) = arrsac
            .model_inliers(&eight_point, matches.iter().copied())
            .unwrap();
        let inliers = inliers.iter().map(|&ix| matches[ix]);
        info!("inliers after sample consensus: {}", inliers.clone().len());
        let residual_average = inliers
            .clone()
            .map(|m| essential.residual(&m).abs())
            .sum::<f32>()
            / inliers.len() as f32;
        info!(
            "inlier residual average after sample consensus: {}",
            residual_average
        );

        // Perform chirality test to determine which essential matrix is correct.
        let (pose, inliers) = essential
            .pose_solver()
            .solve_unscaled_inliers(
                MinimalSquareReprojectionErrorTriangulator::new(),
                inliers.clone().take(8),
            )
            .unwrap();
        info!("inliers after chirality test: {}", inliers.len());

        // Filter outlier matches based on reprojection error.
        let inliers: Vec<_> = matches
            .iter()
            .cloned()
            .filter(|m| {
                // Get the reprojection error in focal length.
                let error_in_focal_length = pinhole::average_pose_reprojection_error(
                    *pose,
                    *m,
                    MinimalSquareReprojectionErrorTriangulator::new(),
                )
                .unwrap();
                // Then convert it to pixels.
                let error_in_pixels = error_in_focal_length * net_focal;

                debug!("error in pixels: {}", error_in_pixels);

                error_in_pixels < opt.reprojection_threshold
            })
            .collect();
        info!(
            "inliers after reprojection error filtering: {}",
            inliers.clone().len()
        );
        let mean_reprojection_error = inliers
            .iter()
            .copied()
            .map(|m| {
                // Get the reprojection error in focal length.
                let error_in_focal_length = pinhole::average_pose_reprojection_error(
                    *pose,
                    m,
                    MinimalSquareReprojectionErrorTriangulator::new(),
                )
                .unwrap();
                // Then convert it to pixels.
                error_in_focal_length * net_focal
            })
            .sum::<f64>()
            / inliers.len() as f64;
        info!(
            "mean reprojection error after reprojection error filtering: {}",
            mean_reprojection_error
        );
        if let Some(outpath) = opt.output.as_ref() {
            let points: Vec<Point3<f64>> = inliers
                .iter()
                .copied()
                .map(|FeatureMatch(a, b)| {
                    MinimalSquareReprojectionErrorTriangulator::new()
                        .triangulate_relative(pose.0, a, b)
                        .unwrap()
                        .0
                })
                .collect();
            export::export(std::fs::File::create(outpath).unwrap(), points);
        }
        info!("rotation: {:?}", pose.rotation.angle());
        prev = next;
        info!("end frame");
    }
}

fn features_stream(opt: &Opt) -> Receiver<(Vec<akaze::KeyPoint>, Vec<Descriptor>)> {
    let (tx, rx) = sync_channel(5);

    let opt = opt.clone();

    std::thread::spawn(move || {
        for path in &opt.images {
            tx.send(kps_descriptors(path, &opt)).unwrap();
        }
    });

    rx
}

fn kps_descriptors(path: impl AsRef<Path>, opt: &Opt) -> (Vec<akaze::KeyPoint>, Vec<Descriptor>) {
    akaze::Akaze::new(opt.akaze_threshold)
        .extract_path(path)
        .unwrap()
}

fn matching(a_descriptors: &[Descriptor], b_descriptors: &[Descriptor]) -> Vec<(usize, usize)> {
    a_descriptors
        .iter()
        .map(|a| {
            b_descriptors
                .iter()
                .map(|b| a.distance(&b))
                .enumerate()
                .min_by_key(|&(_, d)| d)
                .unwrap()
        })
        .collect::<Vec<_>>()
}
