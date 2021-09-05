use image::{DynamicImage};
use cv_sift::{SimpleRGBAImage, image_pyramid, SIFTConfig};

fn test_scale_space_pyramid_given_image(in_path: &str, out_path: &str) {

    let image: DynamicImage = DynamicImage::ImageRgba8(image::open(in_path).unwrap().to_rgba8());
    let simple_img = SimpleRGBAImage::<f64>::from_dynamic(&image);

    let result = image_pyramid(&simple_img, SIFTConfig::new());

    for (row_idx, row) in result.iter().enumerate() {
        for (entry_idx, _entry) in row.iter().enumerate() {
            let _path = format!("{}/oct_{}_level_{}.png", out_path, row_idx, entry_idx);
            _entry.save(&_path).unwrap();
        }
    }
    // dbg!(simple_img.shape());

    // let simple_img_base_img = base_image(&simple_img, 1.6, 0.5);
    // dbg!(simple_img_base_img.shape());

    // println!("{:?}", simple_img_base_img);
    // simple_img_base_img.as_view().save(out_path).unwrap();
}


pub fn main() {
    let inp = "/home/aalekh/Documents/projects/cv-rust/res/box.png";
    let out = "/home/aalekh/Documents/projects/cv-rust/res/pyramid/";
    test_scale_space_pyramid_given_image(inp, out);
}