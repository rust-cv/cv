use derive_more::{Deref, DerefMut};
use image::{DynamicImage, ImageBuffer, Luma, Pixel};
use imageproc::filter::separable_filter_equal;
use log::*;
use ndarray::{azip, s, Array2, ArrayView2, ArrayViewMut2};
use nshare::{MutNdarray2, RefNdarray2};
use std::f32;

/// The image type we use in this library.
///
/// This is simply a wrapper around a contiguous f32 vector. A reader might
/// question why we opted for this approach, instead of using the image
/// crate's image type, and in fact I would typically err on the side
/// of of avoiding premature optimization and re-using existing code.
/// I tried just using the image crate's types with f32 as a
/// template argument. All operations were approximately 40% slower. That
/// implementation is in the history of this repository if you're curious.
///
/// The below traits have been violated in various parts of this crate,
/// with some image operations applying directly to the buffer. This,
/// again, ended up being a necessary optimization. Using iterators
/// to perform image filters sped them up in some cases by a factor of
/// 2. Unfortunately this makes the resulting code a bit less readable.
///
/// We continue to use the image crate for loading and saving images.
///
/// There exists the imageproc crate at the time of this writing, that
/// have existing implementations of generalized image convolutions,
/// Gaussian blur, and image resizing. I re-implemented these things here
/// because the image crate versions are missing some key optimizations
/// like using a separable filter, and using the filters implemented
/// here ended up speeding up everything a lot.
#[derive(Debug, Clone, Deref, DerefMut)]
pub struct GrayFloatImage(pub ImageBuffer<Luma<f32>, Vec<f32>>);

impl GrayFloatImage {
    /// Create a unit float image from the image crate's DynamicImage type.
    ///
    /// # Arguments
    /// * `input_image` - the input image.
    /// # Return value
    /// An image with pixel values between 0 and 1.
    pub fn from_dynamic(input_image: &DynamicImage) -> Self {
        Self(match input_image.grayscale() {
            DynamicImage::ImageLuma8(gray_image) => {
                info!(
                    "Loaded a {} x {} 8-bit image",
                    input_image.width(),
                    input_image.height()
                );
                ImageBuffer::from_fn(gray_image.width(), gray_image.height(), |x, y| {
                    Luma([f32::from(gray_image[(x, y)][0]) / 255f32])
                })
            }
            DynamicImage::ImageLuma16(gray_image) => {
                info!(
                    "Loaded a {} x {} 16-bit image",
                    input_image.width(),
                    input_image.height()
                );
                ImageBuffer::from_fn(gray_image.width(), gray_image.height(), |x, y| {
                    Luma([f32::from(gray_image[(x, y)][0]) / 65535f32])
                })
            }
            DynamicImage::ImageRgb32F(float_image) => {
                info!(
                    "Loaded a {} x {} 32-bit RGB float image",
                    input_image.width(),
                    input_image.height()
                );
                ImageBuffer::from_fn(float_image.width(), float_image.height(), |x, y| {
                    Luma([float_image[(x, y)].to_luma()[0]])
                })
            }
            DynamicImage::ImageRgba32F(float_image) => {
                info!(
                    "Loaded a {} x {} 32-bit RGBA float image",
                    input_image.width(),
                    input_image.height()
                );
                ImageBuffer::from_fn(float_image.width(), float_image.height(), |x, y| {
                    Luma([float_image[(x, y)].to_luma()[0]])
                })
            }
            _ => panic!("DynamicImage::grayscale() returned unexpected type"),
        })
    }

    pub fn from_array2(arr: Array2<f32>) -> Self {
        Self(
            ImageBuffer::from_raw(arr.dim().1 as u32, arr.dim().0 as u32, arr.into_raw_vec())
                .expect("raw vector didn't have enough pixels for the image"),
        )
    }

    pub fn ref_array2(&self) -> ArrayView2<f32> {
        self.0.ref_ndarray2()
    }

    pub fn mut_array2(&mut self) -> ArrayViewMut2<f32> {
        self.0.mut_ndarray2()
    }

    pub fn zero_array(&self) -> Array2<f32> {
        Array2::zeros((self.height(), self.width()))
    }

    pub fn width(&self) -> usize {
        self.0.width() as usize
    }

    pub fn height(&self) -> usize {
        self.0.height() as usize
    }

    pub fn new(width: usize, height: usize) -> Self {
        Self(ImageBuffer::from_pixel(
            width as u32,
            height as u32,
            Luma([0.0]),
        ))
    }

    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.get_pixel(x as u32, y as u32)[0]
    }

    pub fn put(&mut self, x: usize, y: usize, pixel_value: f32) {
        self.put_pixel(x as u32, y as u32, Luma([pixel_value]));
    }

    pub fn half_size(&self) -> Self {
        let width = self.width() / 2;
        let height = self.height() / 2;
        let mut half = Array2::zeros((height, width));

        // 2x2 tiles for everything except the bottom, right, and bottom right corner.
        azip!((
            out in &mut half,
            window in self.ref_array2().slice(s![..height * 2, ..width * 2]).exact_chunks((2, 2)),
        ) {
            *out = window.sum() * 0.25;
        });

        // Bottom (only if not divisible!)
        if height * 2 != self.height() {
            azip!((
                out in half.slice_mut(s![-1.., ..]),
                window in self.ref_array2().slice(s![-1.., ..width * 2]).exact_chunks((1, 2)),
            ) {
                *out = window.sum() * 0.5;
            });
        }

        // Right (only if not divisible!)
        if width * 2 != self.width() {
            azip!((
                out in half.slice_mut(s![.., -1..]),
                window in self.ref_array2().slice(s![..height * 2, -1..]).exact_chunks((2, 1)),
            ) {
                *out = window.sum() * 0.5;
            });
        }

        // Bottom right corner (only if not divisible by both)
        if width * 2 != self.width() && height * 2 != self.height() {
            // This is overkill, but it just copies the bottom right corner pixel.
            azip!((
                out in half.slice_mut(s![-1.., -1..]),
                &pixel in self.ref_array2().slice(s![-1.., -1..]),
            ) {
                *out = pixel;
            });
        }

        Self::from_array2(half)
    }
}

/// The Gaussian function.
///
/// # Arguments
/// * `x` - the offset.
/// * `r` - sigma.
/// # Return value
/// The kernel value at x.
fn gaussian(x: f32, r: f32) -> f32 {
    ((2.0 * f32::consts::PI).sqrt() * r).recip() * (-x.powi(2) / (2.0 * r.powi(2))).exp()
}

/// Generate a Gaussina kernel.
///
/// # Arguments
/// * `r` - sigma.
/// * `kernel_size` - The size of the kernel.
/// # Return value
/// The kernel (a vector).
fn gaussian_kernel(r: f32, kernel_size: usize) -> Vec<f32> {
    let mut kernel = vec![0f32; kernel_size];
    let half_width = (kernel_size / 2) as i32;
    let mut sum = 0f32;
    for i in -half_width..=half_width {
        let val = gaussian(i as f32, r);
        kernel[(i + half_width) as usize] = val;
        sum += val;
    }
    for val in kernel.iter_mut() {
        *val /= sum;
    }
    kernel
}

/// Perform Gaussian blur on an image.
///
/// # Arguments
/// * `r` - sigma.
/// * `kernel_size` - The size of the kernel.
/// # Return value
/// The resulting image after the filter was applied.
pub fn gaussian_blur(image: &GrayFloatImage, r: f32) -> GrayFloatImage {
    assert!(r > 0.0, "sigma must be > 0.0");
    let kernel_radius = (2.0 * r).ceil() as usize;
    let kernel_size = kernel_radius * 2 + 1;
    let kernel = gaussian_kernel(r, kernel_size);
    GrayFloatImage(separable_filter_equal(image, &kernel))
}

#[cfg(test)]
mod tests {
    use super::gaussian_kernel;
    #[test]
    fn gaussian_kernel_correct() {
        // test against known correct kernel
        let kernel = gaussian_kernel(3.0, 7);
        let known_correct_kernel = vec![
            0.1062_8852,
            0.1403_2133,
            0.1657_7007,
            0.1752_4014,
            0.1657_7007,
            0.1403_2133,
            0.1062_8852,
        ];
        for it in kernel.iter().zip(known_correct_kernel.iter()) {
            let (i, j) = it;
            assert!(f32::abs(*i - *j) < 0.0001);
        }
    }
}
