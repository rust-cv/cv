use crate::evolution::EvolutionStep;
use crate::image::{GrayFloatImage, ImageFunctions};
use nalgebra::Vector4;

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
    let Ld: &mut GrayFloatImage = &mut evolution_step.Lt;
    let c: &GrayFloatImage = &evolution_step.Lflow;
    let Lstep: &mut GrayFloatImage = &mut evolution_step.Lstep;
    let w = Lstep.width();

    // Significant positions and ranges
    let xbegin = 0;
    let xmiddle = 1..Lstep.width() - 1;
    let xend = Lstep.width() - 1;
    let ybegin = 0;
    let ymiddle = 1..Lstep.height() - 1;
    let yend = Lstep.height() - 1;

    // Middle diffusion
    for y in ymiddle.clone() {
        let mut Ld_yn = Ld.iter();
        // Up 1 right 1
        let mut Ld_yn_i = Ld_yn.nth(w * (y - 1) + 1).unwrap();

        let mut Ld_yp = Ld.iter();
        // Down 1 right 1
        let mut Ld_yp_i = Ld_yp.nth(w * (y + 1) + 1).unwrap();

        let mut Ld_xn = Ld.iter();
        // Middle
        let mut Ld_xn_i = Ld_xn.nth(w * y).unwrap();

        let mut Ld_x = Ld.iter();
        // Right 1
        let mut Ld_x_i = Ld_x.nth(w * y + 1).unwrap();

        let mut Ld_xp = Ld.iter();
        // Right 2
        let mut Ld_xp_i = Ld_xp.nth(w * y + 2).unwrap();

        let mut c_yn = c.iter();
        // Up 1 right 1
        let mut c_yn_i = c_yn.nth(w * (y - 1) + 1).unwrap();

        let mut c_yp = c.iter();
        // Down 1 right 1
        let mut c_yp_i = c_yp.nth(w * (y + 1) + 1).unwrap();

        let mut c_xn = c.iter();
        // Middle
        let mut c_xn_i = c_xn.nth(w * y).unwrap();

        let mut c_x = c.iter();
        // Right 1
        let mut c_x_i = c_x.nth(w * y + 1).unwrap();

        let mut c_xp = c.iter();
        // Right 2
        let mut c_xp_i = c_xp.nth(w * y + 2).unwrap();

        let slice = &mut (***Lstep)[(w * y + 1)..(w * y + w - 1)];
        for Lstep_x_i in slice.iter_mut() {
            let x_pos = (c_x_i + c_xp_i) * (Ld_xp_i - Ld_x_i);
            let x_neg = (c_xn_i + c_x_i) * (Ld_x_i - Ld_xn_i);
            let y_pos = (c_x_i + c_yp_i) * (Ld_yp_i - Ld_x_i);
            let y_neg = (c_yn_i + c_x_i) * (Ld_x_i - Ld_yn_i);
            *Lstep_x_i = 0.5 * (step_size as f32) * (x_pos - x_neg + y_pos - y_neg);

            c_x_i = c_x.next().unwrap();
            c_xp_i = c_xp.next().unwrap();
            c_xn_i = c_xn.next().unwrap();
            c_yp_i = c_yp.next().unwrap();
            c_yn_i = c_yn.next().unwrap();

            Ld_x_i = Ld_x.next().unwrap();
            Ld_xp_i = Ld_xp.next().unwrap();
            Ld_xn_i = Ld_xn.next().unwrap();
            Ld_yp_i = Ld_yp.next().unwrap();
            Ld_yn_i = Ld_yn.next().unwrap();
        }
    }
    // First row
    for x in xmiddle.clone() {
        let x_pos = eval(c, Ld, x, ybegin, [0, 1, 1, 0], [0, 0, 0, 0]);
        let y_pos = eval(c, Ld, x, ybegin, [0, 0, 0, 0], [0, 1, 1, 0]);
        let x_neg = eval(c, Ld, x, ybegin, [-1, 0, 0, -1], [0, 0, 0, 0]);
        Lstep.put(
            x,
            ybegin,
            0.5 * (step_size as f32) * (x_pos - x_neg + y_pos),
        );
    }
    {
        let x_pos = eval(c, Ld, xbegin, ybegin, [0, 1, 1, 0], [0, 0, 0, 0]);
        let y_pos = eval(c, Ld, xbegin, ybegin, [0, 0, 0, 0], [0, 1, 1, 0]);
        Lstep.put(xbegin, ybegin, 0.5 * (step_size as f32) * (x_pos + y_pos));
    }
    {
        let y_pos = eval(c, Ld, xend, ybegin, [0, 0, 0, 0], [0, 1, 1, 0]);
        let x_neg = eval(c, Ld, xend, ybegin, [-1, 0, 0, -1], [0, 0, 0, 0]);
        Lstep.put(xend, ybegin, 0.5 * (step_size as f32) * (-x_neg + y_pos));
    }
    // Last row
    for x in xmiddle.clone() {
        let x_pos = eval(c, Ld, x, yend, [0, 1, 1, 0], [0, 0, 0, 0]);
        let y_pos = eval(c, Ld, x, yend, [0, 0, 0, 0], [0, -1, -1, 0]);
        let x_neg = eval(c, Ld, x, yend, [-1, 0, 0, -1], [0, 0, 0, 0]);
        Lstep.put(x, yend, 0.5 * (step_size as f32) * (x_pos - x_neg + y_pos));
    }
    {
        let x_pos = eval(c, Ld, xbegin, yend, [0, 1, 1, 0], [0, 0, 0, 0]);
        let y_pos = eval(c, Ld, xbegin, yend, [0, 0, 0, 0], [0, -1, -1, 0]);
        Lstep.put(xbegin, yend, 0.5 * (step_size as f32) * (x_pos + y_pos));
    }
    {
        let y_pos = eval(c, Ld, xend, yend, [0, 0, 0, 0], [0, -1, -1, 0]);
        let x_neg = eval(c, Ld, xend, yend, [-1, 0, 0, -1], [0, 0, 0, 0]);
        Lstep.put(xend, yend, 0.5 * (step_size as f32) * (-x_neg + y_pos));
    }
    // First and last columns
    for y in ymiddle {
        {
            let x_pos = eval(c, Ld, xbegin, y, [0, 1, 1, 0], [0, 0, 0, 0]);
            let y_pos = eval(c, Ld, xbegin, y, [0, 0, 0, 0], [0, 1, 1, 0]);
            let y_neg = eval(c, Ld, xbegin, y, [0, 0, 0, 0], [-1, 0, 0, -1]);
            Lstep.put(
                xbegin,
                y,
                0.5 * (step_size as f32) * (x_pos + y_pos - y_neg),
            );
        }
        {
            let y_pos = eval(c, Ld, xend, y, [0, 0, 0, 0], [0, 1, 1, 0]);
            let x_neg = eval(c, Ld, xend, y, [-1, 0, 0, -1], [0, 0, 0, 0]);
            let y_neg = eval(c, Ld, xend, y, [0, 0, 0, 0], [-1, 0, 0, -1]);
            Lstep.put(xend, y, 0.5 * (step_size as f32) * (-x_neg + y_pos - y_neg));
        }
    }

    let mut Lstep_iter = Lstep.iter();
    for Ld_iter in Ld.iter_mut() {
        *Ld_iter += Lstep_iter.next().unwrap();
    }
}

/// Convenience method for calculating x_pos and x_neg that is more compact
#[allow(non_snake_case)]
#[inline(always)]
pub fn eval(
    c: &GrayFloatImage,
    Ld: &GrayFloatImage,
    x: usize,
    y: usize,
    plus_x: impl Into<Vector4<i32>>,
    plus_y: impl Into<Vector4<i32>>,
) -> f32 {
    let plus_x = plus_x.into();
    let plus_y = plus_y.into();
    let set_x = plus_x + Vector4::from_element(x as i32);
    let set_y = plus_y + Vector4::from_element(y as i32);
    // If we access past the upper bounds of image the image class will assert
    let c_components = set_x
        .rows_range(0..2)
        .zip_map(&set_y.rows_range(0..2), |x, y| {
            c.get(x as usize, y as usize)
        });
    let ld_components = set_x
        .rows_range(2..4)
        .zip_map(&set_y.rows_range(2..4), |x, y| {
            Ld.get(x as usize, y as usize)
        });
    (c_components[0] + c_components[1]) * (ld_components[0] - ld_components[1])
}
