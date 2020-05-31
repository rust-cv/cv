mod export;
mod slam;

use cv::{
    camera::pinhole::{self, CameraIntrinsics, CameraIntrinsicsK1Distortion},
    consensus::Arrsac,
    estimate::EightPoint,
    feature::akaze,
    geom::MinimalSquareReprojectionErrorTriangulator,
    Bearing, BitArray, CameraModel, CameraPose, Consensus, FeatureMatch, Model,
    TriangulatorRelative,
};

use cv::nalgebra::{Point2, Point3, Vector2};

use log::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use slam::*;
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

    let calibrate = |(kps, ds): (Vec<akaze::KeyPoint>, Vec<BitArray<64>>)| {
        kps.into_iter()
            .zip(ds)
            .map(|(kp, d)| (intrinsics.calibrate(kp), d))
            .collect()
    };

    let net_focal = (opt.x_focal.powi(2) + opt.y_focal.powi(2)).sqrt();

    // Create a channel that will produce features in another parallel thread.
    let features = features_stream(&opt);

    // Initialize a new vSLAM reconstruction.
    let mut slam = VSlam::new();
    // Add the camera intrinsics.
    slam.add_feed(intrinsics);

    // Add all of the frames.
    for features in features.into_iter().map(calibrate) {
        slam.add_frame(Frame { feed: 0, features });
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
