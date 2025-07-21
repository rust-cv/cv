//! Note: This code came directly from the imageproc code base because it was private code.
//! I could not find an alternative implementation for the image crate. This will likely
//! go away once we find a library that includes bicubic sampling, or if the image
//! crate itself includes it.

use conv::ValueInto;
use image::{GenericImageView, Pixel};
use imageproc::definitions::{Clamp, Image};

fn blend_cubic<P>(px0: &P, px1: &P, px2: &P, px3: &P, x: f32) -> P
where
    P: Pixel,
    P::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let mut outp = *px0;

    for i in 0..(P::CHANNEL_COUNT as usize) {
        let p0 = px0.channels()[i].value_into().unwrap();
        let p1 = px1.channels()[i].value_into().unwrap();
        let p2 = px2.channels()[i].value_into().unwrap();
        let p3 = px3.channels()[i].value_into().unwrap();
        #[rustfmt::skip]
        let pval = p1 + 0.5 * x * (p2 - p0 + x * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + x * (3.0 * (p1 - p2) + p3 - p0)));
        outp.channels_mut()[i] = <P as Pixel>::Subpixel::clamp(pval);
    }

    outp
}

pub fn interpolate_bicubic<P>(image: &Image<P>, x: f32, y: f32, default: P) -> P
where
    P: Pixel + 'static,
    <P as Pixel>::Subpixel: ValueInto<f32> + Clamp<f32>,
{
    let left = x.floor() - 1f32;
    let right = left + 4f32;
    let top = y.floor() - 1f32;
    let bottom = top + 4f32;

    let x_weight = x - (left + 1f32);
    let y_weight = y - (top + 1f32);

    let mut col: [P; 4] = [default, default, default, default];

    let (width, height) = image.dimensions();
    if left < 0f32 || right >= width as f32 || top < 0f32 || bottom >= height as f32 {
        default
    } else {
        for row in top as u32..bottom as u32 {
            let (p0, p1, p2, p3): (P, P, P, P) = unsafe {
                (
                    image.unsafe_get_pixel(left as u32, row),
                    image.unsafe_get_pixel(left as u32 + 1, row),
                    image.unsafe_get_pixel(left as u32 + 2, row),
                    image.unsafe_get_pixel(left as u32 + 3, row),
                )
            };

            let c = blend_cubic(&p0, &p1, &p2, &p3, x_weight);
            col[row as usize - top as usize] = c;
        }

        blend_cubic(&col[0], &col[1], &col[2], &col[3], y_weight)
    }
}
