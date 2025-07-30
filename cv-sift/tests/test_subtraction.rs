use cv_sift::conversion::try_get_rgb_32f;
use cv_sift::ext::ImageExt;
use cv_sift::ImageRgb32F;
use cv_sift::pyramid::subtract;


#[test]
fn subtraction_reflexivity() {
    let ima = image::open("tests/fixtures/sus.png").unwrap();
    let minuend = try_get_rgb_32f(&ima).unwrap();
    let subtrahend = try_get_rgb_32f(&ima).unwrap();

    let result = subtract(&minuend, &subtrahend).unwrap();
    assert!(result.is_zero());
}

#[test]
fn subtraction_of_zero_matrix() {
    let ima = image::open("tests/fixtures/sus.png").unwrap();
    let minuend = try_get_rgb_32f(&ima).unwrap();

    let subtrahend = ImageRgb32F::new(minuend.width(), minuend.height());
    assert!(subtrahend.is_zero());

    let result = subtract(&minuend, &subtrahend).unwrap();
    assert!(result.is_same_as(&minuend));
}
