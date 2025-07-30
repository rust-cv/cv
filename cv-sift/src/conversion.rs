use image::{DynamicImage};
use crate::ImageRgb32F;
use crate::errors::{Result, SIFTError};


pub fn try_get_rgb_32f(img: &DynamicImage) -> Result<ImageRgb32F> {
    match img {

        DynamicImage::ImageRgb8(p)
        => {
            Ok(ImageRgb32F::from_raw(
                p.width(),
                p.height(),
                p.pixels().into_iter().flat_map(|pixel| pixel.0.iter().cloned().map(|v| v as f32)).collect::<Vec<_>>()
            )
                .unwrap())
        },
        DynamicImage::ImageRgb32F(p)
        => {
            Ok(ImageRgb32F::from_raw(
                p.width(),
                p.height(),
                p.pixels().into_iter().flat_map(|pixel| pixel.0.iter().cloned()).collect::<Vec<_>>()
            )
                .unwrap())
        },
        DynamicImage::ImageRgba32F(p)
        => {
            Ok(ImageRgb32F::from_raw(
                p.width(),
                p.height(),
                p.pixels().into_iter().flat_map(|pixel| pixel.0.iter().cloned()).collect::<Vec<_>>()
            )
                .unwrap())
        },
        DynamicImage::ImageLuma16(p) => {
            Ok(ImageRgb32F::from_raw(
                p.width(),
                p.height(),
                p.pixels().into_iter().flat_map(|pixel| [pixel.0[0] as f32; 3]).collect::<Vec<_>>()
            ).unwrap())
        }
        DynamicImage::ImageLumaA16(p) => {
            Ok(ImageRgb32F::from_raw(
                p.width(),
                p.height(),
                p.pixels().into_iter().flat_map(|pixel| [pixel.0[0] as f32; 3]).collect::<Vec<_>>()
            ).unwrap())
        }
        DynamicImage::ImageLuma8(p) => {
            Ok(ImageRgb32F::from_raw(
                p.width(),
                p.height(),
                p.pixels().into_iter().flat_map(|pixel| [pixel.0[0] as f32; 3]).collect::<Vec<_>>()
            ).unwrap())
        }
        DynamicImage::ImageLumaA8(p) => {
            Ok(ImageRgb32F::from_raw(
                p.width(),
                p.height(),
                p.pixels().into_iter().flat_map(|pixel| [pixel.0[0] as f32; 3]).collect::<Vec<_>>()
            ).unwrap())
        }
        _ => {
            Err(SIFTError::Unsupported("ImageRgb32F image can only be one of ImageLuma8, ImageLumaA8, ImageLuma16, ImageLumaA16, ImageImageRgb32F, ImageRgba32F.".to_string()))
        }
    }
}
