use arrsac::Arrsac;
use cv_core::nalgebra::{Point2, Vector2};
use cv_geom::triangulation::LinearEigenTriangulator;
use cv_pinhole::{CameraIntrinsics, CameraIntrinsicsK1Distortion};
use cv_sfm::{VSlam, VSlamSettings};
use eight_point::EightPoint;
use lambda_twist::LambdaTwist;

use log::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use slotmap::Key;
use std::{collections::HashSet, path::PathBuf};
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
    #[structopt(long, default_value = "0.000001")]
    export_maximum_cosine_distance: f64,
    /// Export required observations
    #[structopt(long, default_value = "3")]
    export_robust_minimum_observations: usize,
    /// The x focal length for "The Zurich Urban Micro Aerial Vehicle Dataset"
    #[structopt(long, default_value = "893.39010814")]
    x_focal: f64,
    /// The y focal length for "The Zurich Urban Micro Aerial Vehicle Dataset"
    #[structopt(long, default_value = "898.32648616")]
    y_focal: f64,
    /// The x optical center coordinate for "The Zurich Urban Micro Aerial Vehicle Dataset"
    #[structopt(long, default_value = "951.1310043")]
    x_center: f64,
    /// The y optical center coordinate for "The Zurich Urban Micro Aerial Vehicle Dataset"
    #[structopt(long, default_value = "555.13350077")]
    y_center: f64,
    /// The skew for "The Zurich Urban Micro Aerial Vehicle Dataset"
    #[structopt(long, default_value = "0.0")]
    skew: f64,
    /// The K1 radial distortion for "The Zurich Urban Micro Aerial Vehicle Dataset"
    #[structopt(long, default_value = "-0.28052513")]
    radial_distortion: f64,
    /// If true, will use actual PLY faces for cameras rather than just points.
    #[structopt(long)]
    no_camera_faces: bool,
    /// Output directory for reconstruction PLY files
    #[structopt(short, long)]
    output: Option<PathBuf>,
    /// List of image files
    ///
    /// Default vales are for "The Zurich Urban Micro Aerial Vehicle Dataset"
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
        .map(|file| bincode::deserialize_from(file).expect("failed to deserialize reconstruction"));
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
        Arrsac::new(
            settings.single_view_consensus_threshold,
            Xoshiro256PlusPlus::seed_from_u64(0),
        )
        .initialization_hypotheses(16384)
        .max_candidate_hypotheses(1024)
        .estimations_per_block(256),
        Arrsac::new(
            settings.two_view_consensus_threshold,
            Xoshiro256PlusPlus::seed_from_u64(0),
        )
        .initialization_hypotheses(8192)
        .max_candidate_hypotheses(1024),
        LambdaTwist::new(),
        EightPoint::new(),
        LinearEigenTriangulator::new(),
        Xoshiro256PlusPlus::seed_from_u64(0),
    );

    // Add the feed.
    let feed = vslam.add_feed(intrinsics);

    let mut normalized = HashSet::new();

    // Add the frames.
    for frame_path in &opt.images {
        info!("loading image {}", frame_path.display());
        let image = image::open(frame_path).expect("failed to load image");
        let frame = vslam.add_frame(feed, &image);
        if let Some((reconstruction, _)) = vslam.data.frame(frame).view {
            if normalized.insert(reconstruction) {
                info!("new reconstruction; normalizing reconstruction");
                vslam.normalize_reconstruction(reconstruction);
            }
            info!("exporting reconstruction");
            if let Some(path) = &opt.output {
                if !path.is_dir() {
                    warn!(
                        "output path is not a directory; it must be a directory; skipping export"
                    );
                } else {
                    // Keep track of the old settings
                    let old_settings = vslam.settings;
                    // Set the settings based on the command line arguments for export purposes.
                    vslam.settings.maximum_cosine_distance = opt.export_maximum_cosine_distance;
                    vslam.settings.robust_minimum_observations =
                        opt.export_robust_minimum_observations;
                    let path = path.join(format!(
                        "{}_{}.ply",
                        frame_path.file_name().unwrap().to_str().unwrap(),
                        reconstruction.data().as_ffi()
                    ));
                    vslam.export_reconstruction(reconstruction, &path, !opt.no_camera_faces);
                    info!("exported {}", path.display());
                    // Restore the pre-export settings.
                    vslam.settings = old_settings;
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
}
