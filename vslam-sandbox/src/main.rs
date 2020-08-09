use cv::nalgebra::{Point2, Vector2};
use cv::{
    camera::pinhole::{CameraIntrinsics, CameraIntrinsicsK1Distortion},
    consensus::Arrsac,
    estimate::{EightPoint, LambdaTwist},
    geom::MinSquaresTriangulator,
};
use cv_reconstruction::{VSlam, VSlamSettings};
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
    /// The maximum cosine distance an observation can have to be exported.
    #[structopt(long, default_value = "0.0000001")]
    export_cosine_distance_threshold: f64,
    /// Export bundle adjust and filter iterations.
    #[structopt(long, default_value = "1")]
    export_reconstruction_optimization_iterations: usize,
    /// Export required observations
    #[structopt(long, default_value = "3")]
    export_robust_minimum_observations: usize,
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

    info!("trying to load existing reconstruction data");
    let data = std::fs::File::open(&opt.data)
        .ok()
        .and_then(|file| bincode::deserialize_from(file).ok());
    if data.is_some() {
        info!("loaded existing reconstruction");
    } else {
        info!("used empty reconstruction");
    }
    let data = data.unwrap_or_default();

    let settings = std::fs::File::open(&opt.settings)
        .ok()
        .and_then(|file| serde_json::from_reader(file).ok());
    if settings.is_some() {
        info!("loaded existing settings");
    } else {
        info!("used default settings");
    }
    let settings: VSlamSettings = settings.unwrap_or_default();

    // Create a channel that will produce features in another parallel thread.
    let mut vslam = VSlam::new(
        data,
        settings,
        Arrsac::new(settings.consensus_threshold, Pcg64::from_seed([5; 32])),
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
                vslam.optimize_reconstruction(reconstruction);
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
        // Set the settings based on the command line arguments for export purposes.
        vslam.settings.cosine_distance_threshold = opt.export_cosine_distance_threshold;
        vslam.settings.reconstruction_optimization_iterations =
            opt.export_reconstruction_optimization_iterations;
        let reconstruction = vslam.data.reconstructions().next().unwrap();
        vslam.optimize_reconstruction(reconstruction);
        // Set settings for robustness to control what is rendered.
        vslam.settings.robust_maximum_cosine_distance = opt.export_cosine_distance_threshold;
        vslam.settings.robust_minimum_observations = opt.export_robust_minimum_observations;
        vslam.export_reconstruction(reconstruction, path);
    }
}
