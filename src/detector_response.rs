use crate::derivatives;
use crate::evolution::{Config, EvolutionStep};
use crate::image::{GrayFloatImage, ImageFunctions};

fn compute_multiscale_derivatives_for_evolution(evolution: &mut EvolutionStep, sigma_size: u32) {
    evolution.Lx = derivatives::scharr(&evolution.Lsmooth, true, false, sigma_size);
    evolution.Ly = derivatives::scharr(&evolution.Lsmooth, false, true, sigma_size);
    evolution.Lxx = derivatives::scharr(&evolution.Lx, true, false, sigma_size);
    evolution.Lyy = derivatives::scharr(&evolution.Ly, false, true, sigma_size);
    evolution.Lxy = derivatives::scharr(&evolution.Lx, false, true, sigma_size);
}

fn compute_multiscale_derivatives(evolutions: &mut Vec<EvolutionStep>, options: Config) {
    for evolution in evolutions.iter_mut() {
        // The image decreases in size by a factor which is 2^octave.
        let ratio = 2.0f64.powi(evolution.octave as i32);
        // The scale of the edge filter.
        let sigma_size = f64::round(evolution.esigma * options.derivative_factor / ratio) as u32;
        compute_multiscale_derivatives_for_evolution(evolution, sigma_size);
    }
}

/// Compute the detector response - the determinant of the Hessian - and save the result
/// in the evolutions.
///
/// # Arguments
/// * `evolutions` - The computed evolutions.
/// * `options` - The options
#[allow(non_snake_case)]
pub fn detector_response(evolutions: &mut Vec<EvolutionStep>, options: Config) {
    compute_multiscale_derivatives(evolutions, options);
    for evolution in evolutions.iter_mut() {
        let ratio = f64::powf(2.0, f64::from(evolution.octave));
        let sigma_size = f64::round(evolution.esigma * options.derivative_factor / ratio) as u32;
        let sigma_size_quat = sigma_size * sigma_size * sigma_size * sigma_size;
        let mut Lxx_iter = evolution.Lxx.buffer.iter();
        let mut Lyy_iter = evolution.Lyy.buffer.iter();
        let mut Lxy_iter = evolution.Lxy.buffer.iter();
        evolution.Ldet = GrayFloatImage::new(evolution.Lxx.width(), evolution.Lxx.height());
        for Ldet_iter in evolution.Ldet.buffer.iter_mut() {
            let Lxx_i = Lxx_iter.next().unwrap();
            let Lyy_i = Lyy_iter.next().unwrap();
            let Lxy_i = Lxy_iter.next().unwrap();
            *Ldet_iter = ((Lxx_i * Lyy_i) - (Lxy_i * Lxy_i)) * (sigma_size_quat as f32);
        }
    }
}
