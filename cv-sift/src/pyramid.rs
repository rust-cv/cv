
use crate::{Dimension};
use crate::image::{SimpleRGBImage};
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
///     use cv_sift::{base_image, SimpleRGBImage};
///     use image::io::{Reader};
///     use image::{DynamicImage};
/// 
/// 
///     let img = Reader::open("/home/aalekh/Documents/projects/cv-rust/res/box.png").unwrap().decode().unwrap().to_rgb8();
///     let d_img = DynamicImage::ImageRgb8(img);
///     
///     let box_img = SimpleRGBImage::<f64>::from_dynamic(&d_img);
///     assert_eq!(box_img.shape(), (223, 324));
///     
///     // let base_img = base_image(&box_img, 1.6, 0.5);
///     // assert_eq!(base_img.shape(), (446, 648));
/// 
/// ```
pub fn base_image(
    _img: &SimpleRGBImage<f64>,
    _sigma: f64,
    _assumed_blur: f64
) -> SimpleRGBImage<f64> {

    todo!();

    // let (height, width) = img.shape();
    // println!("{:?}", img);
    // img.to_owned()
    // let raw: ImgProcImage<f64> = img.as_imgproc_image_f64();
    // let res = transform::scale(&raw, 2.0, 2.0, ImgProcScale::Bilinear).unwrap();

    // let final_sigma = {
    //     let sigma_val = sigma * sigma - 4.0 * assumed_blur * assumed_blur;
    //     if sigma_val > 0.01 {
    //         sigma_val.sqrt()
    //     } else {
    //         0.01_f64.sqrt()
    //     }
    // };
    // let neg_two_times_ln_0_005 = 10.59663473309607335491_f64;
    // let ksize = (1.0 + 2.0 * (final_sigma * final_sigma * neg_two_times_ln_0_005 ).sqrt().round()) as u32;

    // let base_img = match ksize % 2 {
    //     0 => filter::gaussian_blur(&res, ksize + 1, final_sigma).unwrap(),
    //     _ => filter::gaussian_blur(&res, ksize, final_sigma).unwrap()
    // };

    // let base_img_as_u8 = base_img.data().iter().map(|x| x.round() as u8).collect::<Vec<u8>>();
    // let out = SimpleRGBImageU8::from_slice(&base_img_as_u8, width * 2, height * 2); 
    // out
}

// pub fn gaussian_images(img: ) {

// }
