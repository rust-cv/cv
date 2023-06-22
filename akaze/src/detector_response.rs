use crate::{derivatives, evolution::EvolutionStep, image::GrayFloatImage, Akaze};
use ndarray::azip;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

impl Akaze {
    fn compute_multiscale_derivatives(&self, evolutions: &mut [EvolutionStep]) {
        let process_evolution = |evolution: &mut EvolutionStep| {
            // The image decreases in size by a factor which is 2^octave.
            let ratio = 2.0f64.powi(evolution.octave as i32);
            // The scale of the edge filter.
            let sigma_size = f64::round(evolution.esigma * self.derivative_factor / ratio) as u32;
            compute_multiscale_derivatives_for_evolution(evolution, sigma_size);
        };
        #[cfg(not(feature = "rayon"))]
        for evolution in evolutions.iter_mut() {
            process_evolution(evolution);
        }
        #[cfg(feature = "rayon")]
        evolutions.into_par_iter().for_each(|evolution| {
            process_evolution(evolution);
        });
    }

    /// Compute the detector response - the determinant of the Hessian - and save the result
    /// in the evolutions.
    ///
    /// # Arguments
    /// * `evolutions` - The computed evolutions.
    /// * `options` - The options
    #[allow(non_snake_case, clippy::suspicious_operation_groupings)]
    pub fn detector_response(&self, evolutions: &mut [EvolutionStep]) {
        self.compute_multiscale_derivatives(evolutions);
        let process_evolution = |evolution: &mut EvolutionStep| {
            let ratio = f64::powi(2.0, evolution.octave as i32);
            let sigma_size = f64::round(evolution.esigma * self.derivative_factor / ratio);
            let sigma_size_quat = sigma_size.powi(4) as f32;
            evolution.Ldet = GrayFloatImage::new(evolution.Lxx.width(), evolution.Lxx.height());
            azip!((
                Ldet in evolution.Ldet.mut_array2(),
                &Lxx in evolution.Lxx.ref_array2(),
                &Lyy in evolution.Lyy.ref_array2(),
                &Lxy in evolution.Lxy.ref_array2(),
            ) {
                *Ldet = (Lxx * Lyy - Lxy * Lxy) * sigma_size_quat;
            });
        };
        #[cfg(not(feature = "rayon"))]
        for evolution in evolutions.iter_mut() {
            process_evolution(evolution);
        }
        #[cfg(feature = "rayon")]
        evolutions.into_par_iter().for_each(|evolution| {
            process_evolution(evolution);
        });
    }
}

fn compute_multiscale_derivatives_for_evolution(evolution: &mut EvolutionStep, sigma_size: u32) {
    #[cfg(not(feature = "rayon"))]
    {
        evolution.Lx = derivatives::scharr_horizontal(&evolution.Lsmooth, sigma_size);
        evolution.Ly = derivatives::scharr_vertical(&evolution.Lsmooth, sigma_size);
        evolution.Lxx = derivatives::scharr_horizontal(&evolution.Lx, sigma_size);
        evolution.Lyy = derivatives::scharr_vertical(&evolution.Ly, sigma_size);
        evolution.Lxy = derivatives::scharr_vertical(&evolution.Lx, sigma_size);
    }
    #[cfg(feature = "rayon")]
    {
        (evolution.Lx, evolution.Ly) = rayon::join(
            || derivatives::scharr_horizontal(&evolution.Lsmooth, sigma_size),
            || derivatives::scharr_vertical(&evolution.Lsmooth, sigma_size),
        );
        (evolution.Lxx, (evolution.Lyy, evolution.Lxy)) = rayon::join(
            || derivatives::scharr_horizontal(&evolution.Lx, sigma_size),
            || {
                rayon::join(
                    || derivatives::scharr_vertical(&evolution.Ly, sigma_size),
                    || derivatives::scharr_vertical(&evolution.Lx, sigma_size),
                )
            },
        )
    }
}
