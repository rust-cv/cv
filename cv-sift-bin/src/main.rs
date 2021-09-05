use image::{DynamicImage};
use cv_sift::{SimpleRGBAImage, base_image};

fn test_base_image(in_path: &str, out_path: &str) {

    let image: DynamicImage = DynamicImage::ImageRgba8(image::open(in_path).unwrap().to_rgba8());

    let simple_img = SimpleRGBAImage::<f64>::from_dynamic(&image);
    dbg!(simple_img.shape());

    let simple_img_base_img = base_image(&simple_img, 1.6, 0.5);
    dbg!(simple_img_base_img.shape());

    println!("{:?}", simple_img_base_img);
    simple_img_base_img.as_view().save(out_path).unwrap();
}


pub fn main() {
    let inp = "/home/aalekh/Documents/projects/cv-rust/res/box.png";
    let out = "/home/aalekh/Documents/projects/cv-rust/res/box_new_base_image.png";
    test_base_image(inp, out);
}