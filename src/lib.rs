mod contrast_factor;
mod derivatives;
mod descriptors;
mod detector_response;
mod evolution;
mod fed_tau;
mod image;
mod keypoint;
mod nonlinear_diffusion;
mod scale_space_extrema;

pub use evolution::Config;
pub use keypoint::{Descriptor, Keypoint};

use crate::image::{gaussian_blur, GrayFloatImage, ImageFunctions};
use ::image::GenericImageView;
use evolution::*;
use log::*;
use std::path::Path;

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
    let mut dst = GrayFloatImage::new(Lx.width(), Lx.height());
    debug_assert!(Lx.width() == Ly.width());
    debug_assert!(Lx.height() == Ly.height());
    let inverse_k: f64 = 1.0f64 / (k * k);
    for y in 0..Lx.height() {
        for x in 0..Lx.width() {
            let Lx_pixel: f64 = f64::from(Lx.get(x, y));
            let Ly_pixel: f64 = f64::from(Ly.get(x, y));
            let dst_pixel: f64 =
                1.0f64 / (1.0f64 + inverse_k * (Lx_pixel * Lx_pixel + Ly_pixel * Ly_pixel));
            dst.put(x, y, dst_pixel as f32);
        }
    }
    dst
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
        evolutions[i].Lstep =
            GrayFloatImage::new(evolutions[i].Lt.width(), evolutions[i].Lt.height());
        for j in 0..evolutions[i].fed_tau_steps.len() {
            trace!("Starting diffusion step.");
            let step_size = evolutions[i].fed_tau_steps[j];
            nonlinear_diffusion::calculate_step(&mut evolutions[i], step_size);
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
) -> (Vec<EvolutionStep>, Vec<Keypoint>, Vec<Descriptor>) {
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
