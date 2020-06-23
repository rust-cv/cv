mod export;
mod vslam;

use cv::{
    camera::pinhole::{CameraIntrinsics, CameraIntrinsicsK1Distortion},
    consensus::Arrsac,
    estimate::{EightPoint, LambdaTwist},
    geom::MinSquaresTriangulator,
};

use cv::nalgebra::{Point2, Vector2};

use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::path::PathBuf;
use structopt::StructOpt;
use vslam::*;

#[derive(StructOpt, Clone)]
#[structopt(name = "vslam-sandbox", about = "A tool for testing vslam algorithms.")]
struct Opt {
    /// The threshold in bits for matching.
    ///
    /// Setting this to a high number disables it.
    #[structopt(short, long, default_value = "64")]
    match_threshold: usize,
    /// The number of points to use in optimization.
    #[structopt(long, default_value = "32")]
    optimization_points: usize,
    /// The number of observances required to export a landmark to PLY.
    #[structopt(long, default_value = "3")]
    minimum_observances: usize,
    /// The number of landmarks to use in bundle adjust.
    #[structopt(long, default_value = "128")]
    bundle_adjust_landmarks: usize,
    /// The threshold for ARRSAC.
    #[structopt(short, long, default_value = "0.001")]
    arrsac_threshold: f64,
    /// The threshold for AKAZE.
    #[structopt(short = "z", long, default_value = "0.001")]
    akaze_threshold: f64,
    /// The threshold for reprojection error in pixels.
    #[structopt(long, default_value = "0.001")]
    cosine_distance_threshold: f64,
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
    #[structopt(long, default_value = "-0.3728755")]
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

    // Fill intrinsics from args.
    let intrinsics = CameraIntrinsicsK1Distortion::new(
        CameraIntrinsics {
            focals: Vector2::new(opt.x_focal, opt.y_focal),
            principal_point: Point2::new(opt.x_center, opt.y_center),
            skew: opt.skew,
        },
        opt.radial_distortion,
    );

    // Create a channel that will produce features in another parallel thread.
    let mut vslam = VSlam::new(
        Arrsac::new(opt.arrsac_threshold, Pcg64::from_seed([5; 32])),
        EightPoint::new(),
        LambdaTwist::new(),
        MinSquaresTriangulator::new(),
        Pcg64::from_seed([5; 32]),
    )
    .akaze_threshold(opt.akaze_threshold)
    .match_threshold(opt.match_threshold)
    .optimization_points(opt.optimization_points)
    .cosine_distance_threshold(opt.cosine_distance_threshold);

    // Add the feed.
    let feed = vslam.insert_feed(intrinsics);

    // Add the frames.
    for path in opt.images {
        let image = image::open(path).expect("failed to load image");
        vslam.insert_frame(feed, &image);
    }

    // vslam.bundle_adjust_highest_observances(opt.bundle_adjust_landmarks);

    // Export the first match
    if let Some(path) = opt.output {
        vslam.export_reconstruction_at(0, opt.minimum_observances, path);
        // vslam.export_covisibility(Pair::new(0, 1), path);
    }
}
