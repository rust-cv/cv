use std::f64::consts::PI;
/// derived from C++ code by Pablo F. Alcantarilla, Jesus Nuevo in the
/// AKAZE library. Notes from original author of the C++ code:
///
/// This code is derived from FED/FJ library from Grewenig et al.,
/// The FED/FJ library allows solving more advanced problems
/// Please look at the following papers for more information about FED:
/// S. Grewenig, J. Weickert, C. Schroers, A. Bruhn. Cyclic Schemes for
/// PDE-Based Image Analysis. Technical Report No. 327, Department of Mathematics,
/// Saarland University, Saarbrücken, Germany, March 2013
/// S. Grewenig, J. Weickert, A. Bruhn. From box filtering to fast explicit diffusion.
/// DAGM, 2010
///
/// This function allocates an array of the least number of time steps such
/// that a certain stopping time for the whole process can be obtained and fills
/// it with the respective FED time step sizes for one cycle
///
/// # Arguments
/// * `T` - Desired process stopping time
/// * `M` - Desired number of cycles
/// * `tau_max` - Stability limit for the explicit scheme
/// * `reordering` - Reordering flag
/// # Return value
/// The vector with the dynamic step sizes
#[allow(non_snake_case)]
pub fn fed_tau_by_process_time(T: f64, M: i32, tau_max: f64, reordering: bool) -> Vec<f64> {
    // All cycles have the same fraction of the stopping time
    fed_tau_by_cycle_time(T / f64::from(M), tau_max, reordering)
}

/// This function allocates an array of the least number of time steps such
/// that a certain stopping time for the whole process can be obtained and fills it
/// it with the respective FED time step sizes for one cycle
///
/// # Arguments
/// * `t` - Desired cycle stopping time
/// * `tau_max` - Stability limit for the explicit scheme
/// * `reordering` - Reordering flag
/// # Return value
/// tau The vector with the dynamic step sizes
#[allow(non_snake_case)]
fn fed_tau_by_cycle_time(t: f64, tau_max: f64, reordering: bool) -> Vec<f64> {
    // number of time steps
    let n = (f64::ceil(f64::sqrt(3.0 * t / tau_max + 0.25) - 0.5f64 - 1.0e-8) + 0.5) as usize;
    // Ratio of t we search to maximal t
    let scale = 3.0 * t / (tau_max * ((n * (n + 1)) as f64));
    fed_tau_internal(n, scale, tau_max, reordering)
}

/// This function allocates an array of time steps and fills it with FED
/// time step sizes
///
/// # Arguments
/// * `n` - Number of internal steps
/// * `scale` - Ratio of t we search to maximal t
/// * `tau_max` - Stability limit for the explicit scheme
/// * `reordering` - Reordering flag
/// # Return value
/// The vector with the dynamic step sizes
fn fed_tau_internal(n: usize, scale: f64, tau_max: f64, reordering: bool) -> Vec<f64> {
    let mut tauh = if reordering {
        vec![0f64; n]
    } else {
        // unsorted tauh
        vec![]
    };
    if n == 0 {
        vec![]
    } else {
        let mut tau: Vec<f64> = vec![0f64; n];
        let c: f64 = 1.0f64 / (4.0f64 * (n as f64) + 2.0f64);
        let d: f64 = scale * tau_max / 2.0f64;
        // Set up originally ordered tau vector
        for k in 0..n {
            let h: f64 = f64::cos(PI * (2.0f64 * (k as f64) + 1.0f64) * c);
            if reordering {
                tauh[k] = d / (h * h);
            } else {
                tau[k] = d / (h * h);
            }
        }
        if reordering {
            // Permute list of time steps according to chosen reordering function

            // Choose kappa cycle with k = n/2
            // This is a heuristic. We can use Leja ordering instead!
            let kappa = n / 2;
            let mut prime = n + 1;
            while !(primal::is_prime(prime as u64)) {
                prime += 1;
            }
            let mut k = 0;
            for t in tau.iter_mut() {
                let mut index = ((k + 1) * kappa) % prime - 1;
                while index >= n {
                    k += 1;
                    index = ((k + 1) * kappa) % prime - 1;
                }
                *t = tauh[index];
                k += 1;
            }
        }
        tau
    }
}
