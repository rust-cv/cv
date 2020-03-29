use crate::image::{
    fill_border, horizontal_filter, vertical_filter, GrayFloatImage, ImageFunctions,
};
use ndarray::{s, Array2};
use ndarray_image::{NdGray, NdImage};

#[cfg(test)]
mod tests {
    use super::{scharr_main_axis_kernel, scharr_off_axis_kernel};
    use approx::relative_eq;

    #[test]
    fn scharr_3x3_main_axis_kernel() {
        let expected_kernel = vec![0.09375f32, 0.3125f32, 0.09375f32];
        let produced_kernel = scharr_main_axis_kernel(1u32);
        assert_eq!(expected_kernel.len(), produced_kernel.len());
        for i in 0..produced_kernel.len() {
            relative_eq!(expected_kernel[i], produced_kernel[i]);
        }
    }

    #[test]
    fn scharr_3x3_off_axis_kernel() {
        let expected_kernel = vec![-1f32, 0f32, 1f32];
        let produced_kernel = scharr_off_axis_kernel(1u32);
        assert_eq!(expected_kernel.len(), produced_kernel.len());
        for i in 0..produced_kernel.len() {
            relative_eq!(expected_kernel[i], produced_kernel[i]);
        }
    }
}

/// Compute the Scharr derivative horizontally
///
/// The implementation of this function is using a separable kernel, for speed.
///
/// # Arguments
/// * `image` - the input image.
/// * `sigma_size` - the scale of the derivative.
///
/// # Return value
/// Output image derivative (an image.)
pub fn scharr_horizontal(image: &GrayFloatImage, sigma_size: u32) -> GrayFloatImage {
    // a separable Scharr kernel
    let k_vertical = scharr_off_axis_kernel(sigma_size);
    let img_horizontal = scharr_main_axis(&image, sigma_size, FilterDirection::Horizontal);
    vertical_filter(&img_horizontal, &k_vertical)
}

/// Compute the Scharr derivative vertically
///
/// The implementation of this function is using a separable kernel, for speed.
///
/// # Arguments
/// * `image` - the input image.
/// * `sigma_size` - the scale of the derivative.
///
/// # Return value
/// Output image derivative (an image.)
pub fn scharr_vertical(image: &GrayFloatImage, sigma_size: u32) -> GrayFloatImage {
    // a separable Scharr kernel
    let k_horizontal = scharr_off_axis_kernel(sigma_size);
    let img_horizontal = horizontal_filter(&image, &k_horizontal);
    scharr_main_axis(&img_horizontal, sigma_size, FilterDirection::Vertical)
}

/// Multiplies and accumulates
fn accumulate_mul_offset(
    accumulator: &mut Array2<f32>,
    source: &NdGray<f32>,
    val: f32,
    border: usize,
    xoff: usize,
    yoff: usize,
) {
    assert_eq!(source.dim(), accumulator.dim());
    let dims = source.dim();
    let mut accumulator =
        accumulator.slice_mut(s![border..dims.0 - border, border..dims.1 - border]);
    accumulator.scaled_add(
        val,
        &source.slice(s![
            yoff..dims.0 + yoff - 2 * border,
            xoff..dims.1 + xoff - 2 * border
        ]),
    );
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum FilterDirection {
    Horizontal,
    Vertical,
}

fn scharr_main_axis(
    image: &GrayFloatImage,
    sigma_size: u32,
    dir: FilterDirection,
) -> GrayFloatImage {
    let input: NdGray<f32> = NdImage(&image.0).into();
    let mut output = Array2::<f32>::zeros([image.height(), image.width()]);
    // Get the border size (we wont fill in this border width of the output).
    let border = sigma_size as usize;
    // Difference between middle and sides of main axis filter.
    let w = 10.0 / 3.0;
    // Side intensity of filter.
    let norm = (1.0 / (2.0 * f64::from(sigma_size) * (w + 2.0))) as f32;
    // Middle intensity of filter.
    let middle = norm * w as f32;

    let mut offsets = [[border, 0], [border, border], [border, 2 * border]];

    if dir == FilterDirection::Horizontal {
        // Swap the offsets if the filter is a horizontal filter.
        for [x, y] in &mut offsets {
            std::mem::swap(x, y);
        }
    }

    // Accumulate the three components.
    accumulate_mul_offset(
        &mut output,
        &input,
        norm,
        border,
        offsets[0][0],
        offsets[0][1],
    );
    accumulate_mul_offset(
        &mut output,
        &input,
        middle,
        border,
        offsets[1][0],
        offsets[1][1],
    );
    accumulate_mul_offset(
        &mut output,
        &input,
        norm,
        border,
        offsets[2][0],
        offsets[2][1],
    );
    let mut output = GrayFloatImage::from_array2(output);
    fill_border(&mut output, border);
    output
}

/// Produce the Scharr kernel for a certain scale, in the off-axis direction.
///
/// # Arguments
/// * `scale` - the scale of the kernel.
///
/// # Return value
/// The kernel.
pub fn scharr_off_axis_kernel(scale: u32) -> Vec<f32> {
    let size = 3 + 2 * (scale - 1) as usize;
    let mut kernel = vec![0f32; size];
    kernel[0] = -1f32;
    kernel[size / 2] = 0f32;
    kernel[size - 1] = 1f32;
    kernel
}

/// Produce the Scharr kernel for a certain scale, in the main-axis direction.
///
/// # Arguments
/// * `scale` - the scale of the kernel.
///
/// # Return value
/// The kernel.
pub fn scharr_main_axis_kernel(scale: u32) -> Vec<f32> {
    let size = 3 + 2 * (scale - 1) as usize;
    let w = 10.0 / 3.0;
    let norm = 1.0 / (2.0 * f64::from(scale) * (w + 2.0));
    let mut kernel = vec![0f32; size];
    kernel[0] = norm as f32;
    kernel[size / 2] = (w * norm) as f32;
    kernel[size - 1] = norm as f32;
    kernel
}
