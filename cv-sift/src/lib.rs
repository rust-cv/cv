mod sift;
mod image;
mod pyramid;

// Expose all utils.
pub mod utils;


pub use crate::sift::{
    blah
};
pub use crate::image::{
    SimpleRGBImageU8,
    Shape
};
pub use crate::pyramid::{
    gaussian_kernels,
    number_of_octaves,
    base_image
};

pub type Dimension = u32;


#[cfg(test)]
mod tests {

    use image::{DynamicImage};
    use image::io::{Reader as ImageReader};
    pub use crate::image::{SimpleRGBImageU8, Shape};
    use crate::pyramid;
    use crate::utils::{assert_similar};


    fn get_box_image() -> SimpleRGBImageU8 {
        let img = ImageReader::open("/home/aalekh/Documents/projects/cv-rust/res/box.png").unwrap().decode().unwrap().to_rgb8();
        let d_img = DynamicImage::ImageRgb8(img);
        let s_img = SimpleRGBImageU8::from_dynamic(&d_img);
        s_img
    }

    #[test]
    fn create_a_simple_image_from_dynamic() {
        get_box_image();
    }

    #[test]
    fn gaussian_kernels_default() {
        let result = pyramid::gaussian_kernels(1.6, 3);
        let expected: [f64; 6] = [
            1.6,
            1.2262735,
            1.54500779,
            1.94658784,
            2.452547,
            3.09001559
        ];
        assert_similar(&result, &expected);
    }

    #[test]
    fn gaussian_kernels_sigma_2_num_intervals_5() {
        let result = pyramid::gaussian_kernels(2.0, 5);
        let expected: [f64; 8] = [
            2.0,
            1.13050062,
            1.2986042,
            1.49170451,
            1.71351851,
            1.9683159,
            2.26100123,
            2.5972084
        ];
        assert_similar(&result, &expected);
    }

    #[test]
    fn gaussian_kernels_sigma_0_num_intervals_1() {
        let result = pyramid::gaussian_kernels(0.0, 1);
        let expected: [f64; 4] = [
            0.0,
            0.0,
            0.0,
            0.0
        ];
        assert_similar(&result, &expected);
    }


    #[test]
    fn number_of_octaves_100_200() {
        let result = pyramid::number_of_octaves(100, 200);
        assert_eq!(result, 6);
    }

    #[test]
    fn number_of_octaves_223_324() {
        let result = pyramid::number_of_octaves(223, 324);
        assert_eq!(result, 7);
    }

    #[test]
    fn number_of_octaves_box_image() {
        let img = get_box_image();
        let result = pyramid::number_of_octaves(img.height, img.width);
        assert_eq!(result, 7);
    }

    #[test]
    fn base_image_for_box_image() {
        let img = get_box_image();
        let og_shape = img.shape();
        let result = pyramid::base_image(&img, 1.6, 0.5);

        let expected_shape = (og_shape.0 * 2, og_shape.1 * 2);
        let obtained_shape = result.shape();

        assert_eq!(expected_shape, obtained_shape);
    }
}
