use cv::nalgebra::{Point, U3};
use rand::SeedableRng;
use std::{path::PathBuf, thread};
// use log::*;
use rand_pcg::Pcg64;

use cv::nalgebra::{Point2, Vector2};
use cv::{
    camera::pinhole::{CameraIntrinsics, CameraIntrinsicsK1Distortion},
    consensus::Arrsac,
    estimate::{EightPoint, LambdaTwist},
    geom::MinSquaresTriangulator,
};
use cv_reconstruction::{VSlam, VSlamSettings};

#[derive(Clone, Debug)] // StructOpt
pub struct VslamInputParameters {
    /// The file where reconstruction data is accumulated.
    ///
    /// If this file doesn't exist, the file will be created when the program finishes.
    data: PathBuf,
    /// The file where settings are specified.
    ///
    /// This is in the format of `cv_reconstruction::VSlamSettings`.
    settings: PathBuf,
    /// The maximum cosine distance an observation can have to be exported.
    export_cosine_distance_threshold: f64,
    /// Export bundle adjust and filter iterations.
    export_reconstruction_optimization_iterations: usize,
    /// Export required observations
    export_robust_minimum_observations: usize,
    /// The x focal length
    x_focal: f64,
    /// The y focal length
    y_focal: f64,
    /// The x optical center coordinate
    x_center: f64,
    /// The y optical center coordinate
    y_center: f64,
    /// The skew
    skew: f64,
    /// The K1 radial distortion
    radial_distortion: f64,

    /// Output PLY file to deposit point cloud
    // output: Option<PathBuf>,

    /// List of image files
    ///
    /// Default vales are for Kitti 2011_09_26 camera 0
    images: Vec<PathBuf>,
}

impl VslamInputParameters {
    pub fn new(
        x_focal: f64,
        y_focal: f64,
        x_center: f64,
        y_center: f64,
        radial_distortion: f64,
        images: Vec<PathBuf>,
        settings: PathBuf,
        //skew: f64,
        //output: Option<PathBuf>,
        //data: PathBuf,
        //export_cosine_distance_threshold: f64,
        //export_reconstruction_optimization_iterations: usize,
        //export_robust_minimum_observations: usize,
    ) -> Self {
        Self {
            x_focal,
            y_focal,
            x_center,
            y_center,
            radial_distortion,
            images,
            settings,
            skew: 0.0f64,
            data: "vslam.cvr".into(),
            export_cosine_distance_threshold: 0.0000001f64,
            export_reconstruction_optimization_iterations: 1usize,
            export_robust_minimum_observations: 3usize,
            //output: None,
        }
    }
}

pub struct VSlamResult {
    pub point_cloud: Vec<(Point<f64, U3>, [u8; 3])>,
}

fn fix_path(path: &PathBuf) -> PathBuf {
    let current_dir = std::env::current_dir().expect("Not able to get the current dir");
    let new_path = current_dir.join("data").join(path);
    new_path
}

pub fn run_vslam_algo(input_parameters: VslamInputParameters)
//-> Result<VSlamResult, ()>
{
    thread::spawn(move || {
        println!("{:?}", input_parameters);

        // Fill intrinsics from args.
        let intrinsics = CameraIntrinsicsK1Distortion::new(
            CameraIntrinsics {
                focals: Vector2::new(input_parameters.x_focal, input_parameters.y_focal),
                principal_point: Point2::new(input_parameters.x_center, input_parameters.y_center),
                skew: input_parameters.skew,
            },
            input_parameters.radial_distortion,
        );

        let data = std::fs::File::open(&input_parameters.data)
            .ok()
            .and_then(|file| bincode::deserialize_from(file).ok());

        if data.is_some() {
            println!("Loaded existing reconstruction");
        } else {
            println!("Used empty reconstruction");
        }

        let data = data.unwrap_or_default();

        let settings_path = fix_path(&input_parameters.settings);

        let settings = std::fs::File::open(settings_path)
            .ok()
            .and_then(|file| serde_json::from_reader(file).ok());

        if settings.is_some() {
            println!("Loaded existing settings");
        } else {
            println!("Used default settings");
        }

        let settings: VSlamSettings = settings.unwrap_or_default();
        println!("{:?}", settings);

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
        for path in &input_parameters.images {
            // TODO: handle absolute/relative images path
            let new_path = fix_path(path);
            let image = image::open(new_path.clone());

            if image.is_err() {
                println!("Failed to load image {:?}", new_path);
                continue;
            } else {
                println!("Valid image path for {:?}", new_path);
            }

            if let Some(reconstruction) = vslam.add_frame(feed, &image.unwrap()) {
                if vslam.data.reconstruction(reconstruction).views.len() >= 3 {
                    vslam.optimize_reconstruction(reconstruction);
                }
            }
        }

        // if !input_parameters.images.is_empty() {
        //     println!("saving the reconstruction data");
        //     if let Ok(file) = std::fs::File::create(input_parameters.data) {
        //         if let Err(e) = bincode::serialize_into(file, &vslam.data) {
        //             println!("unable to save reconstruction data: {}", e);
        //         }
        //     }
        // } else
        {
            println!("reconstruction not modified, so not saving reconstruction data");
        }

        println!("exporting the reconstruction");
        //if let Some(path) = input_parameters.output {
        // Set the settings based on the command line arguments for export purposes.
        vslam.settings.cosine_distance_threshold =
            input_parameters.export_cosine_distance_threshold;

        vslam.settings.reconstruction_optimization_iterations =
            input_parameters.export_reconstruction_optimization_iterations;

        let reconstruction = vslam.data.reconstructions().next().unwrap();

        vslam.optimize_reconstruction(reconstruction);

        // Set settings for robustness to control what is rendered.
        vslam.settings.robust_maximum_cosine_distance =
            input_parameters.export_cosine_distance_threshold;

        vslam.settings.robust_minimum_observations =
            input_parameters.export_robust_minimum_observations;

        //vslam.export_reconstruction(reconstruction, fix_path(&PathBuf::new().join("output_ply.ply")));

        let result = vslam.all_points_and_colors(reconstruction).collect();

        let output = VSlamResult {
            point_cloud: result,
        };

        println!(
            "There is {} points in the point cloud.",
            output.point_cloud.len()
        );

        // Ok(VSlamResult {
        //     point_cloud: result
        // })
    });
    //}
}
