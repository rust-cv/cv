use image::{DynamicImage};
use cv_sift::{SimpleRGBImageU8, base_image, Shape};

fn test_base_image(in_path: &str, out_path: &str) {

    let image: DynamicImage = image::open(in_path).unwrap();

    let simple_img = SimpleRGBImageU8::from_dynamic(&image);
    dbg!(simple_img.shape());

    let simple_img_base_img = base_image(&simple_img, 1.6, 0.5);
    dbg!(simple_img_base_img.shape());

    simple_img_base_img.as_view().save(out_path).unwrap();
}


pub fn main() {
    let inp = "/home/aalekh/Documents/projects/cv-rust/res/box.png";
    let out = "/home/aalekh/Documents/projects/cv-rust/res/box_base_image.png";
    test_base_image(inp, out);
}