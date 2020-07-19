use cv::{
    camera::pinhole::{CameraIntrinsics, CameraIntrinsicsK1Distortion},
    consensus::Arrsac,
    estimate::{EightPoint, LambdaTwist},
    geom::MinSquaresTriangulator,
};

use cv::nalgebra::{Point2, Vector2};

use cv_reconstruction::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt, Clone)]
#[structopt(name = "vslam-sandbox", about = "A tool for testing vslam algorithms")]
struct Opt {
    /// The threshold in bits for matching.
    ///
    /// Setting this to a high number disables it.
    #[structopt(short, long, default_value = "64")]
    match_threshold: usize,
    /// The number of points to use in optimization.
    #[structopt(long, default_value = "8192")]
    optimization_points: usize,
    /// The number of observations required to export a landmark to PLY.
    #[structopt(long, default_value = "3")]
    minimum_observations: usize,
    /// The number of landmarks to use in bundle adjust.
    #[structopt(long, default_value = "32768")]
    bundle_adjust_landmarks: usize,
    /// The number of iterations to run bundle adjust and filtering globally.
    #[structopt(long, default_value = "2")]
    bundle_adjust_filter_iterations: usize,
    /// The threshold for ARRSAC in cosine distance.
    #[structopt(short, long, default_value = "0.001")]
    arrsac_threshold: f64,
    /// The threshold for AKAZE.
    #[structopt(short = "z", long, default_value = "0.001")]
    akaze_threshold: f64,
    /// Loss cutoff.
    ///
    /// Increasing this value causes the residual function to become cosine distance squared of observances
    ///
    /// Decreasing this value causes the tail ends of the cosine distance squared to flatten out, reducing the impact of outliers.
    ///
    /// Make this value around cosine_distance_threshold and arrsac_threshold.
    #[structopt(long, default_value = "0.00002")]
    loss_cutoff: f64,
    /// The threshold for reprojection error in cosine distance on init.
    ///
    /// When this is exceeded, points are filtered from the reconstruction.
    #[structopt(long, default_value = "0.0005")]
    two_view_cosine_distance_threshold: f64,
    /// The threshold for reprojection error in cosine distance.
    ///
    /// When this is exceeded, points are filtered from the reconstruction.
    #[structopt(long, default_value = "0.00001")]
    cosine_distance_threshold: f64,
    /// The minimum reprojection error in cosine distance that all observations must have to merge two landmarks together.
    #[structopt(long, default_value = "0.000002")]
    merge_cosine_distance_threshold: f64,
    /// The threshold for reprojection error in cosine distance when the pointcloud is exported.
    #[structopt(long, default_value = "0.000005")]
    export_cosine_distance_threshold: f64,
    /// The maximum number of times to run two-view optimization.
    #[structopt(long, default_value = "8000")]
    two_view_patience: usize,
    /// The threshold of mean cosine distance standard deviation that terminates optimization.
    ///
    /// The smaller this value is the more accurate the output will be, but it will take longer to execute.
    #[structopt(long, default_value = "0.00000000001")]
    two_view_std_dev_threshold: f64,
    /// The maximum number of landmarks to use for sample consensus of the pose of the camera during tracking.
    ///
    /// This doesn't affect the number of points in the reconstruction, just the points used for tracking.
    /// This has significantly diminishing returns after a certain point.
    #[structopt(long, default_value = "4096")]
    track_landmarks: usize,
    /// The maximum number of times to run many-view optimization.
    #[structopt(long, default_value = "4000")]
    many_view_patience: usize,
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
    #[structopt(long, default_value = "-0.010584")]
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
    .cosine_distance_threshold(opt.cosine_distance_threshold)
    .merge_cosine_distance_threshold(opt.merge_cosine_distance_threshold)
    .two_view_cosine_distance_threshold(opt.two_view_cosine_distance_threshold)
    .two_view_patience(opt.two_view_patience)
    .two_view_std_dev_threshold(opt.two_view_std_dev_threshold)
    .track_landmarks(opt.track_landmarks)
    .many_view_patience(opt.many_view_patience)
    .loss_cutoff(opt.loss_cutoff);

    // Add the feed.
    let feed = vslam.insert_feed(intrinsics);

    // Add the frames.
    for path in opt.images {
        let image = image::open(path).expect("failed to load image");
        if let Some(reconstruction) = vslam.insert_frame(feed, &image) {
            if vslam.reconstruction_view_count(reconstruction) >= 3 {
                for _ in 0..opt.bundle_adjust_filter_iterations {
                    // If there are three or more views, run global bundle-adjust.
                    vslam.bundle_adjust_highest_observances(
                        reconstruction,
                        opt.bundle_adjust_landmarks,
                    );
                    // Filter observations after running bundle-adjust.
                    vslam.filter_observations(reconstruction, opt.cosine_distance_threshold);
                    // Merge landmarks.
                    vslam.merge_nearby_landmarks(reconstruction);
                }
            }
        }
    }

    // Export the first match
    if let Some(path) = opt.output {
        vslam.filter_observations(0, opt.export_cosine_distance_threshold);
        vslam.export_reconstruction(0, opt.minimum_observations, path);
    }
}
