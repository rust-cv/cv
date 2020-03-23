use crate::image::{horizontal_filter, vertical_filter, GrayFloatImage};

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
    let k_horizontal = scharr_main_axis_kernel(sigma_size);
    let k_vertical = scharr_off_axis_kernel(sigma_size);
    let img_horizontal = horizontal_filter(&image, &k_horizontal);
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
    let k_vertical = scharr_main_axis_kernel(sigma_size);
    let k_horizontal = scharr_off_axis_kernel(sigma_size);
    let img_horizontal = horizontal_filter(&image, &k_horizontal);
    vertical_filter(&img_horizontal, &k_vertical)
}

/// Produce the Scharr kernel for a certain scale, in the off-axis direction.
///
/// # Arguments
/// * `scale` - the scale of the kernel.
///
/// # Return value
/// The kernel.
fn scharr_off_axis_kernel(scale: u32) -> Vec<f32> {
    let size = 3 + 2 * (scale - 1) as usize;
    debug_assert!(size >= 3);
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
fn scharr_main_axis_kernel(scale: u32) -> Vec<f32> {
    let size = 3 + 2 * (scale - 1) as usize;
    debug_assert!(size >= 3);
    let w = 10.0 / 3.0;
    let norm = 1.0 / (2.0 * f64::from(scale) * (w + 2.0));
    let mut kernel = vec![0f32; size];
    kernel[0] = norm as f32;
    kernel[size / 2] = (w * norm) as f32;
    kernel[size - 1] = norm as f32;
    kernel
}
