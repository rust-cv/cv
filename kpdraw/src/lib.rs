use akaze::{Akaze, KeyPoint};
use image::{DynamicImage, Rgba};
use imageproc::drawing;

pub fn render_akaze_keypoints(image: &DynamicImage, threshold: f64) -> DynamicImage {
    let akaze = Akaze::new(threshold);
    let (kps, _) = akaze.extract(image);
    let mut image = drawing::Blend(image.to_rgba());
    for KeyPoint { point: (x, y), .. } in kps {
        drawing::draw_cross_mut(&mut image, Rgba([0, 255, 255, 128]), x as i32, y as i32);
    }
    DynamicImage::ImageRgba8(image.0)
}
