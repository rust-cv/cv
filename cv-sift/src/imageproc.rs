use image::{DynamicImage, ImageBuffer, Luma};
use crate::errors::{SIFTError, Result};
use tracing::{
    debug,
    trace,
};
use image::imageops;


pub type GrayImageBuffer = ImageBuffer<Luma<f32>, Vec<f32>>;

pub fn open<P: AsRef<std::path::Path>>(p: P) -> Result<DynamicImage> {
    image::open(p)
    .map_err(|err| SIFTError::Unsupported(err.to_string()))
}


/// Scale the image by a factor of 2 and apply some blur.
/// # Examples
/// ```
///     use cv_sift::base_image;
///     use image::{DynamicImage};
///     
///     let img = image::open("../res/box.png").unwrap();
///     assert_eq!(img.height(), 223);
///     assert_eq!(img.width(), 324);
///     let base_img = base_image(&img, 1.6, 0.5);
///     assert_eq!(base_img.height(), 446);
///     assert_eq!(base_img.width(), 648);
/// 
/// ```
pub fn base_image(
    img: &DynamicImage,
    sigma: f64,
    assumed_blur: f64
) -> DynamicImage {

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
    blurred.grayscale()
}


pub fn try_get_grayscale_buffer(img: &DynamicImage) -> crate::errors::Result<GrayImageBuffer> {
    match img {
        DynamicImage::ImageRgb32F(p) 
        => {
            Ok(GrayImageBuffer::from_raw(
                p.width(), 
                p.height(), 
                p.pixels().into_iter().map(|pixel| pixel.0[0]).collect::<Vec<_>>()
            )
            .unwrap())
        },
        DynamicImage::ImageRgba32F(p) 
        => {
            Ok(GrayImageBuffer::from_raw(
                p.width(), 
                p.height(), 
            p.pixels().into_iter().map(|pixel| pixel.0[0]).collect::<Vec<_>>()
            )
            .unwrap())
        },
        DynamicImage::ImageLuma16(p) => {
            Ok(GrayImageBuffer::from_raw(
                p.width(), 
                p.height(), 
                p.pixels().into_iter().map(|pixel| pixel.0[0] as f32).collect::<Vec<_>>()
            ).unwrap())
        }
        DynamicImage::ImageLumaA16(p) => {
            Ok(GrayImageBuffer::from_raw(
                p.width(), 
                p.height(), 
                p.pixels().into_iter().map(|pixel| pixel.0[0] as f32).collect::<Vec<_>>()
            ).unwrap())
        }
        DynamicImage::ImageLuma8(p) => {
            Ok(GrayImageBuffer::from_raw(
                p.width(), 
                p.height(), 
                p.pixels().into_iter().map(|pixel| pixel.0[0] as f32).collect::<Vec<_>>()
            ).unwrap())
        }
        DynamicImage::ImageLumaA8(p) => {
            Ok(GrayImageBuffer::from_raw(
                p.width(), 
                p.height(), 
                p.pixels().into_iter().map(|pixel| pixel.0[0] as f32).collect::<Vec<_>>()
            ).unwrap())
        }
        _ => {
            Err(crate::errors::SIFTError::Unsupported("Grayscale image can only be one of ImageLuma8, ImageLumaA8, ImageLuma16, ImageLumaA16, ImageRgb32F, ImageRgba32F.".to_string()))
        }
    }
}