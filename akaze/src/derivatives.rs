use crate::image::{separable_filter, GrayFloatImage};

pub fn simple_scharr_horizontal(image: &GrayFloatImage) -> GrayFloatImage {
    // similar to cv::Scharr with xorder=1, yorder=0, scale=1, delta=0
    GrayFloatImage(separable_filter(&image.0, &[-1., 0., 1.], &[3., 10., 3.]))
}

pub fn simple_scharr_vertical(image: &GrayFloatImage) -> GrayFloatImage {
    // similar to cv::Scharr with xorder=0, yorder=1, scale=1, delta=0
    GrayFloatImage(separable_filter(&image.0, &[3., 10., 3.], &[-1., 0., 1.]))
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
    if sigma_size == 1 {
        return simple_scharr_horizontal(image);
    }
    let main_kernel = computer_scharr_kernel(sigma_size, FilterOrder::Main);
    let off_kernel = computer_scharr_kernel(sigma_size, FilterOrder::Off);
    GrayFloatImage(separable_filter(&image.0, &main_kernel, &off_kernel))
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
    if sigma_size == 1 {
        return simple_scharr_vertical(image);
    }
    let main_kernel = computer_scharr_kernel(sigma_size, FilterOrder::Main);
    let off_kernel = computer_scharr_kernel(sigma_size, FilterOrder::Off);
    GrayFloatImage(separable_filter(&image.0, &off_kernel, &main_kernel))
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum FilterOrder {
    Main,
    Off,
}

fn computer_scharr_kernel(sigma_size: u32, order: FilterOrder) -> Vec<f32> {
    // Difference between middle and sides of main axis filter.
    let w = 10.0 / 3.0;
    // Side intensity of filter.
    let norm = (1.0 / (2.0 * f64::from(sigma_size) * (w + 2.0))) as f32;
    // Middle intensity of filter.
    let middle = norm * w as f32;
    // Size of kernel
    let ksize = (3 + 2 * (sigma_size - 1)) as usize;
    let mut kernel = vec![0.0; ksize];
    match order {
        FilterOrder::Main => {
            kernel[0] = -1.0;
            kernel[ksize - 1] = 1.0;
        }
        FilterOrder::Off => {
            kernel[0] = norm;
            kernel[ksize / 2] = middle;
            kernel[ksize - 1] = norm;
        }
    };
    kernel
}
