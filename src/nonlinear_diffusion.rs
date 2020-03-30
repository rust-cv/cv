use crate::evolution::EvolutionStep;
use crate::image::GrayFloatImage;
use log::*;
use ndarray::s;

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
pub fn calculate_step(evolution_step: &mut EvolutionStep, step_size: f64) {
    trace!("diffusion allocating temporaries");
    // Get the ndarray types.
    let mut input = evolution_step.Lt.clone().into_array2();
    let conductivities = evolution_step.Lflow.clone().into_array2();
    trace!("diffusion finished allocating temporaries");

    // Produce the horizontal and vertical conductivity.
    let horizontal_conductivity =
        &conductivities.slice(s![.., 1..]) * &conductivities.slice(s![.., ..-1]);
    let vertical_conductivity =
        &conductivities.slice(s![1.., ..]) * &conductivities.slice(s![..-1, ..]);
    // Produce the horizontal and vertical flow.
    let horizontal_flow = step_size as f32
        * horizontal_conductivity
        * (&input.slice(s![.., 1..]) - &input.slice(s![.., ..-1]));
    let vertical_flow = step_size as f32
        * vertical_conductivity
        * (&input.slice(s![1.., ..]) - &input.slice(s![..-1, ..]));
    // Add the flows into the input
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

    // Replace Lt.
    evolution_step.Lt = GrayFloatImage::from_array2(input);
}
