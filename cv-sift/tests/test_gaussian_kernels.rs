use cv_sift::pyramid::gaussian_kernels;
use cv_sift::utils::assert_similar;
use test_case::test_case;

#[test_case(0.0, 1, vec![0.; 4]; "0 point 0 and 1")]
#[test_case(2., 5, vec![2.0, 1.13050062, 1.2986042, 1.49170451, 1.71351851, 1.9683159, 2.26100123, 2.5972084]; "2 point 0 and 5")]
#[test_case(1.6, 3, vec![1.6, 1.2262735, 1.54500779, 1.94658784, 2.452547, 3.09001559]; "default params")]
fn test_gaussian_kernels(sigma: f64, num_intervals: usize, expected: Vec<f64>) {
    let observed = gaussian_kernels(sigma, num_intervals);
    assert_similar(&observed, &expected);
}