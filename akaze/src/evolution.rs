use crate::{fed_tau, Akaze, GrayFloatImage};
use log::*;

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
    fn new(octave: u32, sublevel: u32, options: &Akaze) -> EvolutionStep {
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
            Ldet: GrayFloatImage::new(0, 0),
            fed_tau_steps: vec![],
        }
    }
}

impl Akaze {
    /// Allocate and calculate prerequisites to the construction of a scale space.
    ///
    /// # Arguments
    /// `width` - The width of the input image.
    /// `height` - The height of the input image.
    /// `options` - The configuration to use.
    pub fn allocate_evolutions(&self, width: u32, height: u32) -> Vec<EvolutionStep> {
        let mut evolutions: Vec<EvolutionStep> = (0..self.max_octave_evolution)
            .filter_map(|octave| {
                let rfactor = 2.0f64.powi(-(octave as i32));
                let level_height = (f64::from(height) * rfactor) as u32;
                let level_width = (f64::from(width) * rfactor) as u32;
                let smallest_dim = std::cmp::min(level_width, level_height);
                // If the smallest dim is less than 40, terminate as we cannot detect features
                // at a scale that small.
                if smallest_dim < 40 {
                    None
                } else {
                    // At a smallest dimension size between 80, only include one sublevel,
                    // as the amount of information in the image is limited.
                    let sublevels = if smallest_dim < 80 {
                        1
                    } else {
                        self.num_sublevels
                    };
                    // Return the sublevels.
                    Some(
                        (0..sublevels)
                            .map(move |sublevel| EvolutionStep::new(octave, sublevel, self)),
                    )
                }
            })
            .flatten()
            .collect();
        // We need to set the tau steps.
        // This is used to produce each evolution.
        // Each tau corresponds to one diffusion time step.
        // In FED (Fast Explicit Diffusion) earlier time steps are smaller
        // because they are more unstable. Once it becomes more stable, the time
        // steps become larger.
        for i in 1..evolutions.len() {
            // Comute the total difference in time between evolutions.
            let ttime = evolutions[i].etime - evolutions[i - 1].etime;
            // Compute the separate tau steps and assign it to the evolution.
            evolutions[i].fed_tau_steps = fed_tau::fed_tau_by_process_time(ttime, 1, 0.25, true);
            debug!(
                "{} steps in evolution {}.",
                evolutions[i].fed_tau_steps.len(),
                i
            );
        }
        evolutions
    }
}
