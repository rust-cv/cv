mod contrast_factor;
mod derivatives;
mod descriptors;
mod detector_response;
mod evolution;
mod fed_tau;
pub mod image;
mod nonlinear_diffusion;
mod scale_space_extrema;

use crate::image::{gaussian_blur, GrayFloatImage};
use ::image::{DynamicImage, ImageResult};
use bitarray::BitArray;
use cv_core::{nalgebra::Point2, ImagePoint};
use evolution::*;
use float_ord::FloatOrd;
use log::*;
use nonlinear_diffusion::pm_g2;
use std::{cmp::Reverse, path::Path};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
struct Instant(f64);
#[cfg(target_arch = "wasm32")]
impl Instant {
    fn now() -> Self {
        #[cfg(feature = "web-sys")]
        {
            Self(
                web_sys::window()
                    .and_then(|w| w.performance())
                    .map_or(0., |p| p.now()),
            )
        }
        #[cfg(not(feature = "web-sys"))]
        {
            Self(0.)
        }
    }
    fn elapsed(&self) -> std::time::Duration {
        #[cfg(feature = "web-sys")]
        {
            std::time::Duration::from_secs_f64((Self::now().0 - self.0) * 0.001)
        }
        #[cfg(not(feature = "web-sys"))]
        {
            std::time::Duration::from_secs(0)
        }
    }
}

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("tried to sample ({x},{y}) out of image bounds ({width}, {height})")]
    SampleOutOfBounds {
        x: isize,
        y: isize,
        width: usize,
        height: usize,
    },
}

/// A point of interest in an image.
/// This pretty much follows from OpenCV conventions.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
/// [`Akaze::new`] can be used to set this threshold and let all other parameters
/// remain default. You can also use the helpers [`Akaze::sparse`] and
/// [`Akaze::dense`]. The default value of `detector_threshold` is `0.001`.
///
#[derive(Debug, Copy, Clone)]
pub struct Akaze {
    /// The maximum number of features to extract
    pub maximum_features: usize,

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
            maximum_features: usize::MAX,
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
            image,
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
            #[cfg(not(feature = "rayon"))]
            {
                evolutions[i].Lx = derivatives::simple_scharr_horizontal(&evolutions[i].Lsmooth);
                trace!("Computing derivative Lx done.");
                evolutions[i].Ly = derivatives::simple_scharr_vertical(&evolutions[i].Lsmooth);
                trace!("Computing derivative Ly done.");
            }
            #[cfg(feature = "rayon")]
            {
                (evolutions[i].Lx, evolutions[i].Ly) = rayon::join(
                    || derivatives::simple_scharr_horizontal(&evolutions[i].Lsmooth),
                    || derivatives::simple_scharr_vertical(&evolutions[i].Lsmooth),
                );
            }
            evolutions[i].Lflow = pm_g2(&evolutions[i].Lx, &evolutions[i].Ly, contrast_factor);
            trace!("Lflow finished.");
            let evolution = &mut evolutions[i];
            for j in 0..evolution.fed_tau_steps.len() {
                trace!("Starting diffusion step.");
                let step_size = evolution.fed_tau_steps[j];
                nonlinear_diffusion::calculate_step(evolution, step_size as f32);
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
    pub fn find_image_keypoints(&self, evolutions: &mut [EvolutionStep]) -> Vec<KeyPoint> {
        let start = Instant::now();
        self.detector_response(evolutions);
        info!("Computed detector response in: {:?}", start.elapsed());
        let start = Instant::now();
        let keypoints = self.detect_keypoints(evolutions);
        info!("Detected keypoints in: {:?}", start.elapsed());
        keypoints
    }

    /// Extract features using the Akaze feature extractor.
    ///
    /// This performs all operations end-to-end.
    ///
    /// # Arguments
    /// * `image` - The input image for which to extract features.
    ///
    /// Returns the keypoints and the descriptors.
    ///
    /// # Example
    /// ```
    /// use std::path::Path;
    /// let akaze = akaze::Akaze::default();
    /// let filename = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/0000000000.png");
    /// let (keypoints, descriptors) = akaze.extract(&image::open(filename).unwrap());
    /// ```
    ///
    pub fn extract(&self, image: &DynamicImage) -> (Vec<KeyPoint>, Vec<BitArray<64>>) {
        let float_image = GrayFloatImage::from_dynamic(image);
        self.extract_from_gray_float_image(&float_image)
    }

    /// Extract features using the Akaze feature extractor.
    ///
    /// This performs all operations end-to-end.
    ///
    /// # Arguments
    /// * `float_image` - The input image for which to extract features, already in float grayscale.
    ///
    /// Returns the keypoints and the descriptors.
    ///
    pub fn extract_from_gray_float_image(
        &self,
        float_image: &GrayFloatImage,
    ) -> (Vec<KeyPoint>, Vec<BitArray<64>>) {
        trace!("Allocating evolutions.");
        let start = Instant::now();
        let mut evolutions =
            self.allocate_evolutions(float_image.0.width(), float_image.0.height());
        info!("Allocated evolutions in: {:?}", start.elapsed());
        trace!("Creating non-linear space.");
        let start = Instant::now();
        self.create_nonlinear_scale_space(&mut evolutions, float_image);
        info!("Created non-linear scale space in: {:?}", start.elapsed());
        trace!("Finding image keypoints.");
        let mut keypoints = self.find_image_keypoints(&mut evolutions);
        trace!("Sorting keypoints by response and truncating the worst keypoints based on the set maximum");
        let start = Instant::now();
        keypoints.sort_unstable_by_key(|kp| Reverse(FloatOrd(kp.response)));
        keypoints.truncate(self.maximum_features);
        info!(
            "Keypoints sorted in: {:?}, {} left.",
            start.elapsed(),
            keypoints.len()
        );
        trace!("Extracting descriptors.");
        let start = Instant::now();
        let (keypoints, descriptors) = self.extract_descriptors(&evolutions, &keypoints);
        info!("Extracted keypoints in: {:?}", start.elapsed());
        trace!("Computing descriptors finished.");
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
    /// let filename = Path::new(env!("CARGO_MANIFEST_DIR")).join("../res/0000000000.png");
    /// let (keypoints, descriptors) = akaze.extract_path(filename).unwrap();
    /// ```
    ///
    pub fn extract_path(
        &self,
        path: impl AsRef<Path>,
    ) -> ImageResult<(Vec<KeyPoint>, Vec<BitArray<64>>)> {
        Ok(self.extract(&::image::open(path)?))
    }
}
