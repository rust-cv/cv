use derive_more::{Deref, DerefMut};
use image::{DynamicImage, ImageBuffer, Luma, Pixel};
use log::*;
use ndarray::{azip, s, Array2, ArrayView2, ArrayViewMut2};
use nshare::{AsNdarray2, AsNdarray2Mut};
use std::f32;
use wide::f32x4;

type GrayImageBuffer = ImageBuffer<Luma<f32>, Vec<f32>>;

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
pub struct GrayFloatImage(pub GrayImageBuffer);

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
            DynamicImage::ImageLumaA8(gray_image) => {
                info!(
                    "Loaded a {} x {} 8-bit image",
                    input_image.width(),
                    input_image.height()
                );
                ImageBuffer::from_fn(gray_image.width(), gray_image.height(), |x, y| {
                    Luma([f32::from(gray_image[(x, y)][0]) / 255f32])
                })
            }
            DynamicImage::ImageLumaA16(gray_image) => {
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
        self.0.as_ndarray2()
    }

    pub fn mut_array2(&mut self) -> ArrayViewMut2<f32> {
        self.0.as_ndarray2_mut()
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

pub fn horizontal_filter(image: &GrayImageBuffer, kernel: &[f32]) -> GrayImageBuffer {
    // Validate kernel size.
    let kernel_size = kernel.len();
    debug_assert!(kernel_size % 2 == 1);
    let kernel_half_size = kernel_size / 2;
    // Prepare output.
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut output = vec![0.0; width * height];
    // Create SIMD kernel, padded with 0.
    let kernel_simd = kernel
        .chunks(4)
        .map(|chunk| {
            let data = [
                #[allow(clippy::get_first)]
                chunk.get(0).copied().unwrap_or(0.0),
                chunk.get(1).copied().unwrap_or(0.0),
                chunk.get(2).copied().unwrap_or(0.0),
                chunk.get(3).copied().unwrap_or(0.0),
            ];
            f32x4::new(data)
        })
        .collect::<Vec<_>>();
    let kernel_simd_size = 4 * (kernel_size + 3) / 4;
    let kernel_simd_extra_elements = kernel_simd_size - kernel_size;
    // Process each row independently.
    let row_in_it = image.as_raw().chunks_exact(width);
    let row_out_it = output.chunks_exact_mut(width);
    let mut scratch = vec![0f32; width + kernel_half_size * 2 + kernel_simd_extra_elements];
    for (row_in, row_out) in row_in_it.zip(row_out_it) {
        // Prefill extended buffer with center and edge values.
        scratch[0..kernel_half_size].fill(row_in[0]);
        scratch[kernel_half_size..kernel_half_size + width].copy_from_slice(row_in);
        scratch[kernel_half_size + width..2 * kernel_half_size + width].fill(row_in[width - 1]);
        scratch[2 * kernel_half_size + width..].fill(0.);
        // Apply kernel.
        scratch
            .windows(kernel_simd_size)
            .zip(row_out)
            .for_each(|(window, output)| {
                *output = window
                    .chunks_exact(4)
                    .map(|chunk| f32x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .zip(kernel_simd.iter())
                    .fold(f32x4::splat(0.), |acc, (a, b)| a.mul_add(*b, acc))
                    .reduce_add()
            });
    }
    GrayImageBuffer::from_raw(width as u32, height as u32, output).unwrap()
}

pub fn vertical_filter(image: &GrayImageBuffer, kernel: &[f32]) -> GrayImageBuffer {
    // Validate kernel size.
    let kernel_size = kernel.len();
    debug_assert!(kernel_size % 2 == 1);
    let kernel_half_size = kernel_size / 2;
    // Prepare output.
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut output = vec![0.0; width * height];
    // Create SIMD kernel, padded with 0.
    let kernel_simd = kernel
        .chunks(4)
        .map(|chunk| {
            let data = [
                #[allow(clippy::get_first)]
                chunk.get(0).copied().unwrap_or(0.0),
                chunk.get(1).copied().unwrap_or(0.0),
                chunk.get(2).copied().unwrap_or(0.0),
                chunk.get(3).copied().unwrap_or(0.0),
            ];
            f32x4::new(data)
        })
        .collect::<Vec<_>>();
    let kernel_simd_size = 4 * (kernel_size + 3) / 4;
    let kernel_simd_extra_elements = kernel_simd_size - kernel_size;
    // We use a scratch buffer of L1 cache width (64 bytes) to optimize memory access.
    const SCRATCH_WIDTH: usize = 16;
    let scratch_height = height + kernel_half_size * 2 + kernel_simd_extra_elements;
    let mut scratch = vec![0f32; SCRATCH_WIDTH * scratch_height];
    let image = image.as_raw();
    for x_s in (0..width).step_by(SCRATCH_WIDTH) {
        // Fill the scratch buffer with the column values.
        // First paddings.
        let x_e: usize = (x_s + SCRATCH_WIDTH).min(width);
        for x in x_s..x_e {
            let scratch_col_start = (x - x_s) * scratch_height;
            for i in 0..kernel_half_size {
                scratch[scratch_col_start + i] = image[x];
            }
            let scratch_end_col_start = scratch_col_start + kernel_half_size + height;
            let image_last_row_start = (height - 1) * width;
            for i in 0..kernel_half_size {
                scratch[scratch_end_col_start + i] = image[image_last_row_start + x];
            }
            for i in 0..kernel_simd_extra_elements {
                scratch[scratch_end_col_start + kernel_half_size + i] = 0.;
            }
        }
        // Then main content.
        for y in 0..height {
            let image_row_start = y * width;
            for x in x_s..x_e {
                scratch[(x - x_s) * scratch_height + y + kernel_half_size] =
                    image[image_row_start + x];
            }
        }
        // Apply kernel.
        let col_count = x_e - x_s;
        scratch
            .chunks(scratch_height)
            .take(col_count)
            .enumerate()
            .for_each(|(dx, col)| {
                let x = x_s + dx;
                col.windows(kernel_simd_size)
                    .enumerate()
                    .for_each(|(y, window)| {
                        let value = window
                            .chunks_exact(4)
                            .map(|chunk| f32x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]))
                            .zip(kernel_simd.iter())
                            .fold(f32x4::splat(0.), |acc, (a, b)| a.mul_add(*b, acc))
                            .reduce_add();
                        output[y * width + x] = value;
                    });
            });
    }
    GrayImageBuffer::from_raw(width as u32, height as u32, output).unwrap()
}

pub fn separable_filter(
    image: &GrayImageBuffer,
    h_kernel: &[f32],
    v_kernel: &[f32],
) -> GrayImageBuffer {
    let h = horizontal_filter(image, h_kernel);
    vertical_filter(&h, v_kernel)
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

/// Generate a Gaussian kernel.
///
/// # Arguments
/// * `r` - sigma.
/// * `kernel_size` - The size of the kernel.
/// # Return value
/// The kernel (a vector).
pub fn gaussian_kernel(r: f32, kernel_size: usize) -> Vec<f32> {
    assert!(kernel_size % 2 == 1, "kernel_size must be odd");
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
    GrayFloatImage(separable_filter(image, &kernel, &kernel))
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

    #[test]
    fn horizontal_filter() {
        let image = image::open("../res/0000000000.png").unwrap();
        let image = super::GrayFloatImage::from_dynamic(&image);
        let kernel = gaussian_kernel(3.0, 7);
        let filtered_ours = super::horizontal_filter(&image.0, &kernel);
        let filtered_imageproc = imageproc::filter::horizontal_filter(&image.0, &kernel);
        imageproc::assert_pixels_eq_within!(filtered_ours, filtered_imageproc, 0.0001);
    }

    #[test]
    fn vertical_filter() {
        let image = image::open("../res/0000000000.png").unwrap();
        let image = super::GrayFloatImage::from_dynamic(&image);
        let kernel = gaussian_kernel(3.0, 7);
        let filtered_ours = super::vertical_filter(&image.0, &kernel);
        let filtered_imageproc = imageproc::filter::vertical_filter(&image.0, &kernel);
        imageproc::assert_pixels_eq_within!(filtered_ours, filtered_imageproc, 0.0001);
    }
}
