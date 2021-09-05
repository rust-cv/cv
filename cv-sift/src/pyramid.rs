use crate::{Dimension};
use crate::image::{SimpleRGBAImage};
use crate::sift::SIFTConfig;
use image::imageops;
use image::{GenericImageView, DynamicImage};
// use imgproc_rs::transform;
// use imgproc_rs::enums::{Scale as ImgProcScale};
// use imgproc_rs::image::{Image as ImgProcImage};
// use imgproc_rs::filter;

/// List of gaussian kernels at which to blur the input image.
/// # Examples
/// ```
///      use cv_sift::{
///         gaussian_kernels,
///      };
///      use cv_sift::utils::{assert_similar};
/// 
///      let kernels = gaussian_kernels(1.6, 3);
///      let expected: [f64; 6] = [
///          1.6,
///          1.2262735,
///          1.54500779,
///          1.94658784,
///          2.452547,
///          3.09001559
///      ];
///      assert_similar(&kernels, &expected);
/// ```
pub fn gaussian_kernels(
    sigma: f64,
    num_intervals: usize
) -> Vec<f64> {

    let images_per_octave = num_intervals + 3;
    let k: f64 = (2.0_f64).powf(1.0 / num_intervals as f64);
    let mut kernels: Vec<f64> = vec![0.0; images_per_octave];
    
    kernels[0] = sigma;

    for idx in 1..images_per_octave {
        let sigma_previous = (k.powf(idx as f64 - 1.0)) * sigma;
        let sigma_total = k * sigma_previous;
        kernels[idx] = (sigma_total.powf(2.0) - sigma_previous.powf(2.0)).sqrt();
    }

    kernels
}

/// Compute the number of octaves in the image pyramid as a function of height and width of the image.
/// # Examples
/// ```
///     use cv_sift::{
///        number_of_octaves
///    };
/// 
///   let num_octaves = number_of_octaves(223, 324);
///   assert_eq!(num_octaves, 7);
/// ```
pub fn number_of_octaves(height: Dimension, width: Dimension) -> u32 {
    if height < width {
        ((height as f64).log2() - 1.0).round() as u32
    } else {
        ((width as f64).log2() - 1.0).round() as u32
    }
}

/// Scale the image by a factor of 2 and apply some blur.
/// # Examples
/// ```
///     use cv_sift::{base_image, SimpleRGBAImage};
///     use image::io::{Reader};
///     use image::{DynamicImage};
/// 
/// 
///     let img = Reader::open("/home/aalekh/Documents/projects/cv-rust/res/box.png").unwrap().decode().unwrap().to_rgba8();
///     let d_img = DynamicImage::ImageRgba8(img);
///     
///     let box_img = SimpleRGBAImage::<f64>::from_dynamic(&d_img);
///     assert_eq!(box_img.shape(), (223, 324));
///     
///     let base_img = base_image(&box_img, 1.6, 0.5);
///     assert_eq!(base_img.shape(), (446, 648));
/// 
/// ```
pub fn base_image(
    img: &SimpleRGBAImage<f64>,
    sigma: f64,
    assumed_blur: f64
) -> SimpleRGBAImage<f64> {

    let (height, width) = img.shape();

    let scaled = imageops::resize(
        &img.as_view(),
        width * 2,
        height * 2,
        imageops::FilterType::Triangle
    );

    let final_sigma = {
        let sigma_val = sigma * sigma - 4.0 * assumed_blur * assumed_blur;
        if sigma_val > 0.01 {
            sigma_val.sqrt()
        } else {
            0.01_f64.sqrt()
        }
    };

    let blurred = imageops::blur(&scaled, final_sigma as f32);
    SimpleRGBAImage::<f64>::from_dynamic(&DynamicImage::ImageRgba8(blurred))
}



/// Given an image, generate the scale space pyramid.
/// # Examples
/// ```
///     use cv_sift::{SimpleRGBAImage, image_pyramid, SIFTConfig};
///     use image::io::{Reader};
///     use image::{DynamicImage};
/// 
///     // Use an absolute path here (for now?)
///     let raw_img = Reader::open("/home/aalekh/Documents/projects/cv-rust/cv-sift/media/box.png")
///     .unwrap()
///     .decode()
///     .unwrap()
///     .to_rgba8();
/// 
///     let d_img = DynamicImage::ImageRgba8(raw_img);
///     let img = SimpleRGBAImage::<f64>::from_dynamic(&d_img);
/// 
///     let out = "/absolute/path/to/a/directory";
///     let result = image_pyramid(&img, SIFTConfig::new());

///     for (row_idx, row) in result.iter().enumerate() {
///         for (entry_idx, entry) in row.iter().enumerate() {
///             let path = format!("{}/oct_{}_level_{}.png", out, row_idx, entry_idx);
///             // entry.save(&path).unwrap();
///             // Uncomment the above line to save the images.
///         }
///     }
/// ```
pub fn image_pyramid(img: &SimpleRGBAImage<f64>, config: SIFTConfig) -> Vec<Vec<SimpleRGBAImage<f64>>> {
    let g_kernels = gaussian_kernels(config.sigma, config.num_intervals);
    let base_img = base_image(&img, config.sigma, config.assumed_blur);
    let num_octaves = number_of_octaves(base_img.height, base_img.width) as u32;

    let mut result: Vec<Vec<SimpleRGBAImage<f64>>> = vec![];

    for vec_blurred in image_pyramid_given_base(&base_img, num_octaves, &g_kernels).iter(){
        let mut vec_blurred_imgs: Vec<SimpleRGBAImage<f64>> = vec![];
        for blurred in vec_blurred.iter() {
            vec_blurred_imgs.push(SimpleRGBAImage::<f64>::from_dynamic(&blurred));
        }
        result.push(vec_blurred_imgs);
    }
    result
}


/// Given a base image, generate a scale space pyramid.
fn image_pyramid_given_base(
    img: &SimpleRGBAImage<f64>,
    num_octaves: u32,
    g_kernels: &Vec<f64>
) -> Vec<Vec<DynamicImage>>{

    let mut img_cp = img.to_owned().as_view().to_owned();
    let mut blurred_images: Vec<Vec<DynamicImage>> = vec![];
    let mut octave_base_image = SimpleRGBAImage::<f64>::zeros_like(img.shape()).as_view().to_owned();

    for _ in 0..num_octaves {
        let mut octave_images: Vec<DynamicImage> = vec![img_cp.clone()];

        for kernel_sigma_idx in 1..g_kernels.len() {
            let kernel_sigma: f64 = g_kernels[kernel_sigma_idx];
            let img_blurred = DynamicImage::ImageRgba8(imageops::blur(&img_cp, kernel_sigma as f32));
            
            if kernel_sigma_idx == g_kernels.len() - 3 {
                octave_base_image = (SimpleRGBAImage::<f64>::from_dynamic(&img_blurred).clone()).as_view().to_owned();
            }
            octave_images.push(img_blurred);
        }

        blurred_images.push(octave_images);
        img_cp = DynamicImage::ImageRgba8(
            imageops::resize(
                &octave_base_image,
                octave_base_image.width() / 2 as u32,
                octave_base_image.height() / 2 as u32,
                imageops::FilterType::Nearest
            )
        );
    }
    blurred_images
}
