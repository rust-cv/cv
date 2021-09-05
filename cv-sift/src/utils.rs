/// Assert two given slices of floats are equal.
/// # Examples:
/// 
/// ### Two exactly equal slices must be equal.
/// ```
/// use cv_sift::utils::assert_similar;
/// 
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![1.0, 2.0, 3.0];
/// assert_similar(&a, &b);
/// ```
/// 
/// ### Elements differ no more than 1e-8.
/// ```
/// use cv_sift::utils::assert_similar;
/// 
/// let a = vec![1.0, 2.0 - 1e-8, 3.0];
/// let b = vec![1.0, 2.0, 3.0 - 1e-9];
/// 
/// assert_similar(&a, &b);
/// ```
pub fn assert_similar(v1: &[f64], v2: &[f64]) {
    assert_eq!(v1.len(), v2.len());
    let result = v1
    .iter()
    .zip(v2.iter())
    .all(|(&f1, &f2)| (f1 - f2).abs() <= 1e-8);
    assert!(result);
}

/// Assert two given slices of floats are not equal.
/// ### Some elements differ more than 1e-8.
/// ```
/// use cv_sift::utils::assert_not_similar;
/// 
/// let a = vec![1.0, 2.0 - 1e-8, 3.0];
/// let b = vec![1.0 - 1e-5, 2.0, 3.0 - 1e-9];
/// 
/// assert_not_similar(&a, &b);
/// ```
pub fn assert_not_similar(v1: &[f64], v2: &[f64]) {
    assert_eq!(v1.len(), v2.len());
    let result = v1
    .iter()
    .zip(v2.iter())
    .any(|(&f1, &f2)| (f1 - f2).abs() > 1e-8);
    assert!(result);
}