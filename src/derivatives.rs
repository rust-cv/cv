use crate::image::{fill_border, GrayFloatImage, ImageFunctions};
use ndarray::{s, Array2};
use ndarray_image::{NdGray, NdImage};

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
    let img_horizontal = scharr_axis(
        &image,
        sigma_size,
        FilterDirection::Horizontal,
        FilterOrder::Main,
    );
    scharr_axis(
        &img_horizontal,
        sigma_size,
        FilterDirection::Vertical,
        FilterOrder::Off,
    )
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
    let img_horizontal = scharr_axis(
        &image,
        sigma_size,
        FilterDirection::Horizontal,
        FilterOrder::Off,
    );
    scharr_axis(
        &img_horizontal,
        sigma_size,
        FilterDirection::Vertical,
        FilterOrder::Main,
    )
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

#[derive(Copy, Clone, Debug, PartialEq)]
enum FilterOrder {
    Main,
    Off,
}

fn scharr_axis(
    image: &GrayFloatImage,
    sigma_size: u32,
    dir: FilterDirection,
    order: FilterOrder,
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

    let mut offsets = match order {
        FilterOrder::Main => vec![
            (norm, [border, 0]),
            (middle, [border, border]),
            (norm, [border, 2 * border]),
        ],
        FilterOrder::Off => vec![(-1.0, [border, 0]), (1.0, [border, 2 * border])],
    };

    if dir == FilterDirection::Horizontal {
        // Swap the offsets if the filter is a horizontal filter.
        for (_, [x, y]) in &mut offsets {
            std::mem::swap(x, y);
        }
    }

    // Accumulate the three components.
    for (val, [x, y]) in offsets {
        accumulate_mul_offset(&mut output, &input, val, border, x, y);
    }
    let mut output = GrayFloatImage::from_array2(output);
    fill_border(&mut output, border);
    output
}
