use cv::feature::akaze::{Akaze, KeyPoint};
use image::{DynamicImage, Rgba};
use imageproc::drawing;

fn main() {
    let src_image = image::open("res/0000000000.png").expect("failed to open image file");

    let threshold = 0.001f64;
    let akaze = Akaze::new(threshold);
    let (key_points, _descriptor) = akaze.extract(&src_image);

    let mut image = drawing::Blend(src_image.to_rgba8());
    for KeyPoint { point: (x, y), .. } in key_points {
        drawing::draw_cross_mut(&mut image, Rgba([0, 255, 255, 128]), x as i32, y as i32);
    }
    let out_image = DynamicImage::ImageRgba8(image.0);
    cv::vis::imgshow(&out_image);
}
