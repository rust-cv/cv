mod contrast_factor;
mod derivatives;
mod descriptors;
mod detector_response;
mod evolution;
mod fed_tau;
mod image;
mod nonlinear_diffusion;
mod scale_space_extrema;

use crate::image::{gaussian_blur, GrayFloatImage};
use ::image::{DynamicImage, GenericImageView, ImageResult};
use bitarray::BitArray;
use cv_core::nalgebra::Point2;
use cv_core::ImagePoint;
use evolution::*;
use log::*;
use nonlinear_diffusion::pm_g2;
use std::path::Path;

/// A point of interest in an image.
/// This pretty much follows from OpenCV conventions.
#[derive(Debug, Clone, Copy)]
pub struct KeyPoint {
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

impl ImagePoint for KeyPoint {
    fn image_point(&self) -> Point2<f64> {
        Point2::new(self.point.0 as f64, self.point.1 as f64)
    }
}

/// Contains the configuration parameters of AKAZE.
///
/// The most important parameter to pay attention to is `detector_threshold`.
/// [`Config::new`] can be used to set this threshold and let all other parameters
/// remain default. You can also use the helpers [`Config::sparse`] and
/// [`Config::dense`]. The default value of `detector_threshold` is `0.001`.
///
#[derive(Debug, Copy, Clone)]
pub struct Akaze {
    /// Default number of sublevels per scale level
    pub num_sublevels: u32,

    /// Maximum octave evolution of the image 2^sigma (coarsest scale sigma units)
    pub max_octave_evolution: u32,

    /// Base scale offset (sigma units)
    pub base_scale_offset: f64,

    /// The initial contrast factor parameter
    pub initial_contrast: f64,

    /// Percentile level for the contrast factor
    pub contrast_percentile: f64,

    /// Number of bins for the contrast factor histogram
    pub contrast_factor_num_bins: usize,

    /// Factor for the multiscale derivatives
    pub derivative_factor: f64,

    /// Detector response threshold to accept point
    pub detector_threshold: f64,

    /// Number of channels in the descriptor (1, 2, 3)
    pub descriptor_channels: usize,

    /// Actual patch size is 2*pattern_size*point.scale
    pub descriptor_pattern_size: usize,
}

impl Akaze {
    /// This convenience constructor is provided for the very common case
    /// that the detector threshold needs to be modified.
    pub fn new(threshold: f64) -> Self {
        Self {
            detector_threshold: threshold,
            ..Default::default()
        }
    }

    /// Create a `Config` that sparsely detects features.
    ///
    /// Uses a threshold of `0.01` (default is `0.001`).
    pub fn sparse() -> Self {
        Self::new(0.01)
    }

    /// Create a `Config` that densely detects features.
    ///
    /// Uses a threshold of `0.0001` (default is `0.001`).
    pub fn dense() -> Self {
        Self::new(0.0001)
    }
}

impl Default for Akaze {
    fn default() -> Akaze {
        Akaze {
            num_sublevels: 4,
            max_octave_evolution: 4,
            base_scale_offset: 1.6f64,
            initial_contrast: 0.001f64,
            contrast_percentile: 0.7f64,
            contrast_factor_num_bins: 300,
            derivative_factor: 1.5f64,
            detector_threshold: 0.001f64,
            descriptor_channels: 3usize,
            descriptor_pattern_size: 10usize,
        }
    }
}

impl Akaze {
    /// A nonlinear scale space performs selective blurring to preserve edges.
    ///
    /// # Arguments
    /// * `evolutions` - The output scale space.
    /// * `image` - The input image.
    fn create_nonlinear_scale_space(
        &self,
        evolutions: &mut Vec<EvolutionStep>,
        image: &GrayFloatImage,
    ) {
        trace!("Creating first evolution.");
        evolutions[0].Lt = gaussian_blur(image, self.base_scale_offset as f32);
        trace!("Gaussian blur finished.");
        evolutions[0].Lsmooth = evolutions[0].Lt.clone();
        debug!(
            "Convolving first evolution with sigma={} Gaussian.",
            self.base_scale_offset
        );
        let mut contrast_factor = contrast_factor::compute_contrast_factor(
            &evolutions[0].Lsmooth,
            self.contrast_percentile,
            1.0f64,
            self.contrast_factor_num_bins,
        );
        trace!("Computing contrast factor finished.");
        debug!(
            "Contrast percentile={}, Num bins={}, Initial contrast factor={}",
            self.contrast_percentile, self.contrast_factor_num_bins, contrast_factor
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
    fn find_image_keypoints(&self, evolutions: &mut Vec<EvolutionStep>) -> Vec<KeyPoint> {
        self.detector_response(evolutions);
        trace!("Computing detector response finished.");
        self.detect_keypoints(evolutions)
    }

    /// Extract features using the Akaze feature extractor.
    ///
    /// This performs all operations end-to-end. The client might be only interested
    /// in certain portions of the process, all of which are exposed in public functions,
    /// but this function can document how the various parts fit together.
    ///
    /// # Arguments
    /// * `image` - The input image for which to extract features.
    /// * `options` - The options for the algorithm. Set this to `None` for default options.
    ///
    /// Returns the keypoints and the descriptors.
    ///
    /// # Example
    /// ```
    /// use std::path::Path;
    /// let akaze = akaze::Akaze::default();
    /// let (keypoints, descriptors) = akaze.extract(&image::open("../res/0000000000.png").unwrap());
    /// ```
    ///
    pub fn extract(&self, image: &DynamicImage) -> (Vec<KeyPoint>, Vec<BitArray<64>>) {
        let float_image = GrayFloatImage::from_dynamic(&image);
        let mut evolutions = self.allocate_evolutions(image.width(), image.height());
        self.create_nonlinear_scale_space(&mut evolutions, &float_image);
        trace!("Finding image keypoints.");
        let keypoints = self.find_image_keypoints(&mut evolutions);
        trace!("Extracting descriptors.");
        let descriptors = self.extract_descriptors(&evolutions, &keypoints);
        trace!("Computing descriptors finished.");
        info!("Extracted {} features", keypoints.len());
        (keypoints, descriptors)
    }

    /// Extract features using the Akaze feature extractor from an image on disk.
    ///
    /// This performs all operations end-to-end. The client might be only interested
    /// in certain portions of the process, all of which are exposed in public functions,
    /// but this function can document how the various parts fit together.
    ///
    /// # Arguments
    /// * `path` - The input image path for which to extract features.
    /// * `options` - The options for the algorithm. Set this to `None` for default options.
    ///
    /// Returns an `ImageResult` of the keypoints and the descriptors.
    ///
    /// # Examples
    /// ```
    /// use std::path::Path;
    /// let akaze = akaze::Akaze::default();
    /// let (keypoints, descriptors) = akaze.extract_path("../res/0000000000.png").unwrap();
    /// ```
    ///
    pub fn extract_path(
        &self,
        path: impl AsRef<Path>,
    ) -> ImageResult<(Vec<KeyPoint>, Vec<BitArray<64>>)> {
        Ok(self.extract(&::image::open(path)?))
    }
}
