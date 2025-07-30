/// Compute the number of octaves in the image pyramid as a function of height and width of the image.
/// # Examples
/// ```
///   use cv_sift::pyramid::number_of_octaves;
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
