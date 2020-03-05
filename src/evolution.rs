use crate::fed_tau;
use crate::image::{GrayFloatImage, ImageFunctions};
use log::*;
use std::path::PathBuf;

#[derive(Debug, Copy, Clone)]
pub struct Config {
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

impl Default for Config {
    fn default() -> Config {
        Config {
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

#[derive(Debug)]
#[allow(non_snake_case)]
pub struct EvolutionStep {
    /// Evolution time
    pub etime: f64,
    /// Evolution sigma. For linear diffusion t = sigma^2 / 2
    pub esigma: f64,
    /// Image octave
    pub octave: u32,
    /// Image sublevel in each octave
    pub sublevel: u32,
    /// Integer sigma. For computing the feature detector responses
    pub sigma_size: u32,
    /// Evolution image
    pub Lt: GrayFloatImage,
    /// Smoothed image
    pub Lsmooth: GrayFloatImage,
    /// First order spatial derivative
    pub Lx: GrayFloatImage,
    /// First order spatial derivatives
    pub Ly: GrayFloatImage,
    /// Second order spatial derivative
    pub Lxx: GrayFloatImage,
    /// Second order spatial derivatives
    pub Lyy: GrayFloatImage,
    /// Second order spatial derivatives
    pub Lxy: GrayFloatImage,
    /// Diffusivity image
    pub Lflow: GrayFloatImage,
    /// Evolution step update
    pub Lstep: GrayFloatImage,
    /// Detector response
    pub Ldet: GrayFloatImage,
    /// fed_tau steps
    pub fed_tau_steps: Vec<f64>,
}

impl EvolutionStep {
    /// Construct a new EvolutionStep for a given octave and sublevel
    ///
    /// # Arguments
    /// * `octave` - The target octave.
    /// * `octave` - The target sublevel.
    /// * `options` - The options to use.
    fn new(octave: u32, sublevel: u32, options: Config) -> EvolutionStep {
        let esigma = options.base_scale_offset
            * f64::powf(
                2.0f64,
                f64::from(sublevel) / f64::from(options.num_sublevels) + f64::from(octave),
            );
        let etime = 0.5 * (esigma * esigma);
        EvolutionStep {
            etime,
            esigma,
            octave,
            sublevel,
            sigma_size: esigma.round() as u32,
            Lt: GrayFloatImage::new(0, 0),
            Lsmooth: GrayFloatImage::new(0, 0),
            Lx: GrayFloatImage::new(0, 0),
            Ly: GrayFloatImage::new(0, 0),
            Lxx: GrayFloatImage::new(0, 0),
            Lyy: GrayFloatImage::new(0, 0),
            Lxy: GrayFloatImage::new(0, 0),
            Lflow: GrayFloatImage::new(0, 0),
            Lstep: GrayFloatImage::new(0, 0),
            Ldet: GrayFloatImage::new(0, 0),
            fed_tau_steps: vec![],
        }
    }
}

/// Allocate and calculate prerequisites to the construction of a scale space.
///
/// # Arguments
/// `width` - The width of the input image.
/// `height` - The height of the input image.
/// `options` - The configuration to use.
pub fn allocate_evolutions(width: u32, height: u32, options: Config) -> Vec<EvolutionStep> {
    let mut out_vec: Vec<EvolutionStep> = vec![];
    for i in 0..options.max_octave_evolution {
        let rfactor = 1.0f64 / f64::powf(2.0f64, f64::from(i));
        let level_height = (f64::from(height) * rfactor) as u32;
        let level_width = (f64::from(width) * rfactor) as u32;
        // Smallest possible octave and allow one scale if the image is small
        if (level_width >= 80 && level_height >= 40) || i == 0 {
            for j in 0..options.num_sublevels {
                let evolution_step = EvolutionStep::new(i, j, options);
                out_vec.push(evolution_step);
            }
        } else {
            break;
        }
    }
    for i in 1..out_vec.len() {
        let ttime = out_vec[i].etime - out_vec[i - 1].etime;
        out_vec[i].fed_tau_steps = fed_tau::fed_tau_by_process_time(ttime, 1, 0.25, true);
        debug!(
            "{} steps in evolution {}.",
            out_vec[i].fed_tau_steps.len(),
            i
        );
    }
    out_vec
}

fn build_path(mut destination_dir: PathBuf, path_label: String, idx: usize) -> PathBuf {
    let to_write = format!("{}{:05}.png", path_label, idx);
    destination_dir.push(to_write);
    destination_dir.set_extension(".png");
    destination_dir
}
