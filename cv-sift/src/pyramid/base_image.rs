use std::convert::TryInto;
use image::{DynamicImage, Rgb};
use tracing::{
    trace,
    debug
};
use image::imageops;
use crate::ImageRgb32F;
use crate::errors::Result;
use image::Pixel;
use crate::conversion::try_get_rgb_32f;



/// Scale the image by a factor of 2 and apply some blur.
/// # Examples
/// ```
///     use cv_sift::pyramid::generate_base_image;
///     use image::{DynamicImage};
///
///     let img = image::open("tests/fixtures/box.png").unwrap();
///     assert_eq!(img.height(), 223);
///     assert_eq!(img.width(), 324);
///     let base_img = generate_base_image(&img, 1.6, 0.5).unwrap();
///     assert_eq!(base_img.height(), 446);
///     assert_eq!(base_img.width(), 648);
///
/// ```
pub fn generate_base_image(
    img: &DynamicImage,
    sigma: f64,
    assumed_blur: f64
) -> Result<ImageRgb32F> {
    let (height, width) = (img.height(), img.width());

    trace!(
        height,
        width,
        sigma,
        assumed_blur,
        "Computing base image"
    );

    let scaled = img.resize(
        width * 2,
        height * 2,
        imageops::FilterType::Triangle
    );

    trace!(
        height = height * 2,
        width = width * 2,
        interpolation = "linear",
        "Scaled image."
    );

    let final_sigma = {
        let sigma_val = sigma * sigma - 4.0 * assumed_blur * assumed_blur;
        if sigma_val > 0.01 {
            sigma_val.sqrt()
        } else {
            0.01_f64.sqrt()
        }
    };

    debug!(final_sigma, "Computed final_sigma for blurring.");

    let blurred = scaled.blur(final_sigma as f32);
    try_get_rgb_32f(&blurred)
}


/// Returns a difference of two images.
pub fn subtract(minuend: &ImageRgb32F, subtrahend: &ImageRgb32F) -> Result<ImageRgb32F> {
    assert_eq!(minuend.height(), subtrahend.height());
    assert_eq!(minuend.width(), subtrahend.width());

    let (width, height) = (minuend.width() as u32, minuend.height() as u32);

    let mut result_mat = ImageRgb32F::new(width, height);
    let mut result_mat_pixels = result_mat.pixels_mut();

    for (minuend_pixel, subtrahend_pixel) in minuend.pixels().zip(subtrahend.pixels()){

        let output_pixel: [f32; 3] =
            minuend_pixel
                .channels()
                .iter()
                .zip(
                    subtrahend_pixel
                        .channels()
                        .iter()
                )
                .map(|(minuend_p, subtrahend_p)| {
                    *minuend_p - *subtrahend_p
                })
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap();

        let next_pixel = result_mat_pixels.next().unwrap();
        *next_pixel = Rgb(output_pixel);
    }
    Ok(result_mat)
}


