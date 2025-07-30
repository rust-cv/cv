use crate::ImageRgb32F;
use image::Pixel;

/// How close do we want two floats to be
/// before they're considered equal.
pub const TOLERANCE: f32 = 1e-8;

pub trait ImageExt {
    fn is_zero(&self) -> bool;
    fn is_same_as(&self, other: &Self) -> bool;
}

impl ImageExt for ImageRgb32F {
    fn is_zero(&self) -> bool {
        self
            .pixels()
            .all(
                |pixel|
                    pixel
                        .channels()
                        .iter()
                        .find(|p| (**p).abs() > TOLERANCE)
                        .is_none()
            )
    }

    fn is_same_as(&self, other: &Self) -> bool {

        for (own_pixel, other_pixel) in self.pixels().zip(other.pixels()) {
            for (self_p, other_p) in own_pixel.channels().iter().zip(other_pixel.channels().iter()) {
                if (*self_p - *other_p).abs() > TOLERANCE {
                    return false;
                }
            }
        }
        true
    }
}
