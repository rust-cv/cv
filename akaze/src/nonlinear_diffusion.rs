use crate::{EvolutionStep, GrayFloatImage};
use ndarray::{azip, s, Array2};

/// This function performs a scalar non-linear diffusion step.
///
/// # Arguments
/// * `Ld` - Output image in the evolution
/// * `c` - Conductivity image. The function c is a scalar value that depends on the gradient norm
/// * `Lstep` - Previous image in the evolution
/// * `step_size` - The step size in time units
/// Forward Euler Scheme 3x3 stencil
/// dL_by_ds = d(c dL_by_dx)_by_dx + d(c dL_by_dy)_by_dy
#[allow(non_snake_case)]
pub fn calculate_step(evolution_step: &mut EvolutionStep, step_size: f32) {
    // Get the ndarray types.
    let mut input = evolution_step.Lt.mut_array2();
    let conductivities = evolution_step.Lflow.ref_array2();
    let dim = input.dim();
    // Horizontal flow.
    let mut horizontal_flow = Array2::<f32>::zeros((dim.0, dim.1 - 1));
    azip!((
        flow in &mut horizontal_flow,
        &a in input.slice(s![.., ..-1]),
        &b in input.slice(s![.., 1..]),
        &ca in conductivities.slice(s![.., ..-1]),
        &cb in conductivities.slice(s![.., 1..]),
    ) {
        *flow = 0.5 * step_size * (ca + cb) * (b - a);
    });
    // Vertical flow.
    let mut vertical_flow = Array2::<f32>::zeros((dim.0 - 1, dim.1));
    azip!((
        flow in &mut vertical_flow,
        &a in input.slice(s![..-1, ..]),
        &b in input.slice(s![1.., ..]),
        &ca in conductivities.slice(s![..-1, ..]),
        &cb in conductivities.slice(s![1.., ..]),
    ) {
        *flow = 0.5 * step_size * (ca + cb) * (b - a);
    });

    // Left
    input
        .slice_mut(s![.., ..-1])
        .zip_mut_with(&horizontal_flow, |acc, &i| *acc += i);
    // Right
    input
        .slice_mut(s![.., 1..])
        .zip_mut_with(&horizontal_flow, |acc, &i| *acc -= i);
    // Up
    input
        .slice_mut(s![..-1, ..])
        .zip_mut_with(&vertical_flow, |acc, &i| *acc += i);
    // Down
    input
        .slice_mut(s![1.., ..])
        .zip_mut_with(&vertical_flow, |acc, &i| *acc -= i);
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
pub fn pm_g2(Lx: &GrayFloatImage, Ly: &GrayFloatImage, k: f64) -> GrayFloatImage {
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
