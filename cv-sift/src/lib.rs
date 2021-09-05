mod sift;
mod image;
mod pyramid;

// Expose all utils.
pub mod utils;


pub use crate::sift::{
    SIFTConfig
};
pub use crate::image::{
    SimpleRGBAImage,
    ApplyAcrossChannels
};

pub use crate::pyramid::{
    gaussian_kernels,
    number_of_octaves,
    base_image,
    image_pyramid
};

pub type Dimension = u32;


#[cfg(test)]
mod tests {

    use image::{DynamicImage};
    use image::io::{Reader as ImageReader};
    use crate::image::{SimpleRGBAImage, ApplyAcrossChannels};
    use crate::pyramid;
    use crate::utils::{assert_similar};
    use crate::sift::{SIFTConfig};
    use nalgebra::{DMatrix};


    fn get_box_image_f64() -> SimpleRGBAImage<f64> {
        let img = ImageReader::open("/home/aalekh/Documents/projects/cv-rust/res/box.png")
        .unwrap()
        .decode()
        .unwrap()
        .to_rgba8();
        let d_img = DynamicImage::ImageRgba8(img);
        SimpleRGBAImage::<f64>::from_dynamic(&d_img)
    }

    fn get_box_image_u8() -> SimpleRGBAImage<u8> {
        let img = ImageReader::open("/home/aalekh/Documents/projects/cv-rust/res/box.png")
        .unwrap()
        .decode()
        .unwrap()
        .to_rgba8();
        let d_img = DynamicImage::ImageRgba8(img);
        SimpleRGBAImage::<u8>::from_dynamic(&d_img)
    }

    fn do_nothing(m: &DMatrix<f64>) -> DMatrix<f64> {
        m.clone()
    }
    fn do_nothing_mut(m: &mut DMatrix<f64>) {
        m.add_scalar_mut(0.0);
    }

    #[test]
    fn create_a_simple_image_from_dynamic() {
        get_box_image_u8();
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
        let img = get_box_image_u8();
        let result = pyramid::number_of_octaves(img.height, img.width);
        assert_eq!(result, 7);
    }



    #[test]
    fn apply_channels_mut_do_nothing() {
        let expected = get_box_image_f64();
        let mut result = get_box_image_f64();
        result.apply_channels_mut(&mut do_nothing_mut);
        assert!(result.is_same_as(&expected));
    }

    #[test]
    fn apply_channels_do_nothing() {
        let expected = get_box_image_f64();
        let result = get_box_image_f64().apply_channels(&do_nothing);
        assert!(result.is_same_as(&expected));
    }

    #[test]
    fn base_image_for_box_image() {
        let img = get_box_image_f64();
        let og_shape = img.shape();
        let result = pyramid::base_image(&img, 1.6, 0.5);

        let expected_shape = (og_shape.0 * 2, og_shape.1 * 2);
        let obtained_shape = result.shape();

        assert_eq!(expected_shape, obtained_shape);

        // println!("{:?}", result);
    }


    #[test]
    fn zeros_like_400_800() {
        let height = 400;
        let width = 800;
        let result = SimpleRGBAImage::<f64>::zeros_like((height, width));
        assert_eq!(result.shape(), (height, width));
        assert!(
            result.is_same_as(
                &SimpleRGBAImage::<f64>::from_slice(
                    &vec![0.0; 400 * 800 * 4],
                    width,
                    height
                )
            )
        );
    }

    #[test]
    fn image_pyramid_for_box_image() {
        let img = get_box_image_f64();
        let out = "/home/aalekh/Documents/projects/cv-rust/cv-sift/media/box_pyramid";

        let result = pyramid::image_pyramid(&img, SIFTConfig::new());

        for (row_idx, row) in result.iter().enumerate() {
            for (entry_idx, _entry) in row.iter().enumerate() {
                let _path = format!("{}/oct_{}_level_{}.png", out, row_idx, entry_idx);
                // _entry.save(&_path).unwrap();
            }
        }
    }
}
