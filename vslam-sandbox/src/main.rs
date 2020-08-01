use cv::nalgebra::{Point2, Vector2};
use cv::{
    camera::pinhole::{CameraIntrinsics, CameraIntrinsicsK1Distortion},
    consensus::Arrsac,
    estimate::{EightPoint, LambdaTwist},
    geom::MinSquaresTriangulator,
};
use cv_reconstruction::VSlam;
use log::*;
use rand::SeedableRng;
use rand_pcg::Pcg64;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt, Clone)]
#[structopt(name = "vslam-sandbox", about = "A tool for testing vslam algorithms")]
struct Opt {
    /// The file where reconstruction data is accumulated.
    ///
    /// If this file doesn't exist, the file will be created when the program finishes.
    #[structopt(short, long, default_value = "vslam.cvr")]
    data: PathBuf,
    /// The file where settings are specified.
    ///
    /// This is in the format of `cv_reconstruction::VSlamSettings`.
    #[structopt(short, long, default_value = "vslam-settings.json")]
    settings: PathBuf,
    /// The threshold for ARRSAC in cosine distance.
    #[structopt(long, default_value = "0.001")]
    arrsac_threshold: f64,
    /// The number of observations required to export a landmark.
    #[structopt(long, default_value = "5")]
    export_minimum_observations: usize,
    /// The maximum cosine distance an observation can have to be exported.
    #[structopt(long, default_value = "0.0000001")]
    export_cosine_distance_threshold: f64,
    /// The minimum cosine distance between any pair of observations required on a landmark which is exported.
    ///
    /// Use this to reduce the outliers when exporting.
    #[structopt(long, default_value = "0.0001")]
    export_minimum_cosine_distance: f64,
    /// Export bundle adjust and filter iterations.
    #[structopt(long, default_value = "1")]
    export_bundle_adjust_filter_iterations: usize,
    /// The number of iterations to run bundle adjust and filtering globally.
    #[structopt(long, default_value = "3")]
    bundle_adjust_filter_iterations: usize,
    /// The number of landmarks to use in bundle adjust.
    #[structopt(long, default_value = "32768")]
    bundle_adjust_landmarks: usize,
    /// The threshold for reprojection error in cosine distance.
    ///
    /// When this is exceeded, points are filtered from the reconstruction.
    #[structopt(long, default_value = "0.00001")]
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

    info!("loading existing reconstruction data");
    let vslam_data = std::fs::File::open(&opt.data)
        .ok()
        .and_then(|file| {
            let data = bincode::deserialize_from(file).ok();
            if data.is_some() {
                info!("loaded existing data");
            }
            data
        })
        .unwrap_or_default();

    info!("loading existing settings");
    let vslam_settings = std::fs::File::open(&opt.settings)
        .ok()
        .and_then(|file| {
            let settings = bincode::deserialize_from(file).ok();
            if settings.is_some() {
                info!("loaded existing settings");
            }
            settings
        })
        .unwrap_or_default();

    // Create a channel that will produce features in another parallel thread.
    let mut vslam = VSlam::new(
        vslam_data,
        vslam_settings,
        Arrsac::new(opt.arrsac_threshold, Pcg64::from_seed([5; 32])),
        EightPoint::new(),
        LambdaTwist::new(),
        MinSquaresTriangulator::new(),
        Pcg64::from_seed([5; 32]),
    );

    // Add the feed.
    let init_reconstruction = vslam.data.reconstructions().next();
    let feed = vslam.add_feed(intrinsics, init_reconstruction);

    // Add the frames.
    for path in &opt.images {
        let image = image::open(path).expect("failed to load image");
        if let Some(reconstruction) = vslam.add_frame(feed, &image) {
            if vslam.data.reconstruction(reconstruction).views.len() >= 3 {
                for _ in 0..opt.bundle_adjust_filter_iterations {
                    // If there are three or more views, run global bundle-adjust.
                    vslam.bundle_adjust_highest_observations(
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

    if !opt.images.is_empty() {
        info!("saving the reconstruction data");
        if let Ok(file) = std::fs::File::create(opt.data) {
            if let Err(e) = bincode::serialize_into(file, &vslam.data) {
                error!("unable to save reconstruction data: {}", e);
            }
        }
    } else {
        info!("reconstruction not modified, so not saving reconstruction data");
    }

    info!("exporting the reconstruction");
    if let Some(path) = opt.output {
        let reconstruction = vslam.data.reconstructions().next().unwrap();
        vslam.filter_observations(reconstruction, opt.export_cosine_distance_threshold);
        for _ in 0..opt.export_bundle_adjust_filter_iterations {
            // If there are three or more views, run global bundle-adjust.
            vslam.bundle_adjust_highest_observations(reconstruction, opt.bundle_adjust_landmarks);
            // Filter observations after running bundle-adjust.
            vslam.filter_observations(reconstruction, opt.export_cosine_distance_threshold);
            // Merge landmarks.
            vslam.merge_nearby_landmarks(reconstruction);
        }
        vslam.export_reconstruction(reconstruction, path);
    }
}
