use image::{DynamicImage};
use cv_sift::image::{SimpleRGBImageU8, SimpleRGBImageF64};

// pub mod image;


pub fn main() {

    let image: DynamicImage = image::open("/home/aalekh/Documents/projects/cv-rust/res/0000000002.jpeg").unwrap();

    let simple_img1 = SimpleRGBImageU8::from_dynamic(&image);
    let simple_img2 = SimpleRGBImageF64::from_dynamic(&image);
    
    let s_img_ref1 = &simple_img1;
    let s_img_ref2 = &simple_img2;

    s_img_ref1.as_view().save("/home/aalekh/Documents/projects/cv-rust/res/0000000002_u8.png").unwrap();
    s_img_ref2.as_view().save("/home/aalekh/Documents/projects/cv-rust/res/0000000002_f64.png").unwrap();

}