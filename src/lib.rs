mod contrast_factor;
mod derivatives;
mod descriptors;
mod detector_response;
mod evolution;
mod fed_tau;
mod image;
mod nonlinear_diffusion;
mod scale_space_extrema;

pub use evolution::Config;

use crate::image::{gaussian_blur, GrayFloatImage};
use ::image::GenericImageView;
use evolution::*;
use log::*;
use ndarray::azip;
use std::path::Path;

use cv_core::nalgebra::Point2;
use cv_core::ImagePoint;
use space::{Bits512, Hamming};

/// A point of interest in an image.
/// This pretty much follows from OpenCV conventions.
#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    /// The horizontal coordinate in a coordinate system is
    /// defined s.t. +x faces right and starts from the top
    /// of the image.
    /// the vertical coordinate in a coordinate system is defined
    /// s.t. +y faces toward the bottom of an image and starts
    /// from the left side of the image.
    pub point: (f32, f32),
    /// The magnitude of response from the detector.
    pub response: f32,

    /// The radius defining the extent of the keypoint, in pixel units
    pub size: f32,

    /// The level of scale space in which the keypoint was detected.
    pub octave: usize,

    /// A classification ID
    pub class_id: usize,

    /// The orientation angle
    pub angle: f32,
}

impl ImagePoint for Keypoint {
    fn image_point(&self) -> Point2<f64> {
        Point2::new(self.point.0 as f64, self.point.1 as f64)
    }
}

/// This function computes the Perona and Malik conductivity coefficient g2
/// g2 = 1 / (1 + dL^2 / k^2)
///
/// # Arguments
/// * `Lx` - First order image derivative in X-direction (horizontal)
/// * `Ly` - First order image derivative in Y-direction (vertical)
/// * `k` - Contrast factor parameter
/// # Return value
/// Output image
#[allow(non_snake_case)]
fn pm_g2(Lx: &GrayFloatImage, Ly: &GrayFloatImage, k: f64) -> GrayFloatImage {
    assert!(Lx.width() == Ly.width());
    assert!(Lx.height() == Ly.height());
    let inverse_k = (1.0f64 / (k * k)) as f32;
    let mut conductivities = Lx.zero_array();
    azip!((
        c in &mut conductivities,
        &x in Lx.ref_array2(),
        &y in Ly.ref_array2(),
    ) {
        *c = 1.0 / (1.0 + inverse_k * (x * x + y * y));
    });
    GrayFloatImage::from_array2(conductivities)
}

/// A nonlinear scale space performs selective blurring to preserve edges.
///
/// # Arguments
/// * `evolutions` - The output scale space.
/// * `image` - The input image.
/// * `options` - The options to use.
fn create_nonlinear_scale_space(
    evolutions: &mut Vec<EvolutionStep>,
    image: &GrayFloatImage,
    options: Config,
) {
    trace!("Creating first evolution.");
    evolutions[0].Lt = gaussian_blur(image, options.base_scale_offset as f32);
    trace!("Gaussian blur finished.");
    evolutions[0].Lsmooth = evolutions[0].Lt.clone();
    debug!(
        "Convolving first evolution with sigma={} Gaussian.",
        options.base_scale_offset
    );
    let mut contrast_factor = contrast_factor::compute_contrast_factor(
        &evolutions[0].Lsmooth,
        options.contrast_percentile,
        1.0f64,
        options.contrast_factor_num_bins,
    );
    trace!("Computing contrast factor finished.");
    debug!(
        "Contrast percentile={}, Num bins={}, Initial contrast factor={}",
        options.contrast_percentile, options.contrast_factor_num_bins, contrast_factor
    );
    for i in 1..evolutions.len() {
        trace!("Creating evolution {}.", i);
        if evolutions[i].octave > evolutions[i - 1].octave {
            evolutions[i].Lt = evolutions[i - 1].Lt.half_size();
            trace!("Half-sizing done.");
            contrast_factor *= 0.75;
            debug!(
                "New image size: {}x{}, new contrast factor: {}",
                evolutions[i].Lt.width(),
                evolutions[i].Lt.height(),
                contrast_factor
            );
        } else {
            evolutions[i].Lt = evolutions[i - 1].Lt.clone();
        }
        evolutions[i].Lsmooth = gaussian_blur(&evolutions[i].Lt, 1.0f32);
        trace!("Gaussian blur finished.");
        evolutions[i].Lx = derivatives::scharr_horizontal(&evolutions[i].Lsmooth, 1);
        trace!("Computing derivative Lx done.");
        evolutions[i].Ly = derivatives::scharr_vertical(&evolutions[i].Lsmooth, 1);
        trace!("Computing derivative Ly done.");
        evolutions[i].Lflow = pm_g2(&evolutions[i].Lx, &evolutions[i].Ly, contrast_factor);
        trace!("Lflow finished.");
        for j in 0..evolutions[i].fed_tau_steps.len() {
            trace!("Starting diffusion step.");
            let step_size = evolutions[i].fed_tau_steps[j];
            nonlinear_diffusion::calculate_step(&mut evolutions[i], step_size as f32);
            trace!("Diffusion step finished with step size {}", step_size);
        }
    }
}

/// Find image keypoints using the Akaze feature extractor.
///
/// # Arguments
/// * `input_image` - An image from which to extract features.
/// * `options` the options for the algorithm.
/// # Return Value
/// The resulting keypoints.
///
fn find_image_keypoints(evolutions: &mut Vec<EvolutionStep>, options: Config) -> Vec<Keypoint> {
    detector_response::detector_response(evolutions, options);
    trace!("Computing detector response finished.");
    scale_space_extrema::detect_keypoints(evolutions, options)
}

/// Extract features using the Akaze feature extractor.
///
/// This performs all operations end-to-end. The client might be only interested
/// in certain portions of the process, all of which are exposed in public functions,
/// but this function can document how the various parts fit together.
///
/// # Arguments
/// * `input_image_path` - The input image for which to extract features.
/// * `output_features_path` - The output path to which to write an output JSON file.
/// * `options` The options for the algorithm.
///
/// # Return value
/// * The evolutions of the process. Can be used for further analysis or visualization, or ignored.
/// * The keypoints at which features occur.
/// * The descriptors that were computed.
///
/// # Examples
/// ```no_run
/// extern crate akaze;
/// use std::path::Path;
/// let options = akaze::Config::default();
/// let (_evolutions, keypoints, descriptors) =
///     akaze::extract_features(
///       "test-data/1.jpg",
///       options);
/// ```
///
pub fn extract_features(
    input_image_path: impl AsRef<Path>,
    options: Config,
) -> (Vec<EvolutionStep>, Vec<Keypoint>, Vec<Hamming<Bits512>>) {
    let input_image = ::image::open(input_image_path).unwrap();
    let float_image = GrayFloatImage::from_dynamic(&input_image);
    info!(
        "Loaded a {} x {} image",
        input_image.width(),
        input_image.height()
    );
    let mut evolutions =
        evolution::allocate_evolutions(input_image.width(), input_image.height(), options);
    create_nonlinear_scale_space(&mut evolutions, &float_image, options);
    trace!("Creating scale space finished.");
    let keypoints = find_image_keypoints(&mut evolutions, options);
    let descriptors = descriptors::extract_descriptors(&evolutions, &keypoints, options);
    trace!("Computing descriptors finished.");
    (evolutions, keypoints, descriptors)
}
