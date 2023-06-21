
/// List of gaussian kernels at which to blur the input image.
/// # Examples
/// ```
///      use cv_sift::{
///         gaussian_kernels,
///      };
///      use cv_sift::{assert_similar};
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
    for (idx, item) in kernels.iter_mut().enumerate().take(images_per_octave).skip(1) {
        let sigma_previous = (k.powf(idx as f64 - 1.0)) * sigma;
        let sigma_total = k * sigma_previous;
        *item = (sigma_total.powf(2.0) - sigma_previous.powf(2.0)).sqrt();

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
pub fn number_of_octaves(height: u32, width: u32) -> u32 {
    if height < width {
        ((height as f64).log2() - 1.0).round() as u32
    } else {
        ((width as f64).log2() - 1.0).round() as u32
    }
}
