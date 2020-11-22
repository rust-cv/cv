use image::{DynamicImage, GenericImageView, Rgba};
use imageproc::drawing;
use rand::Rng;

fn main() {
    let src_image = image::open("../../res/0000000000.png").expect("failed to open image file");
    let mut rng = rand::thread_rng();
    let mut canvas = drawing::Blend(src_image.to_rgba8());
    for _ in 0..50 {
        let x = rng.gen_range(0, src_image.width()) as i32;
        let y = rng.gen_range(0, src_image.height()) as i32;
        drawing::draw_cross_mut(&mut canvas, Rgba([0, 255, 255, 128]), x, y);
    }

    let out_img = DynamicImage::ImageRgba8(canvas.0);
    cv::vis::imgshow(&out_img);
}
