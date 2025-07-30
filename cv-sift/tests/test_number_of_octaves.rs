use cv_sift::pyramid::number_of_octaves;
use test_case::test_case;

#[test_case(223, 324, 7; "223x324")]
#[test_case(100, 200, 6; "100x200")]
fn test_number_of_octaves(height: u32, width: u32, expected: u32) {
    assert_eq!(number_of_octaves(height, width), expected);
}
