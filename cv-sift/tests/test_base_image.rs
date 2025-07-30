use test_case::test_case;


#[test_case("testdata/box.png"; "box_")]
fn test_base_image(in_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let img = cv_sift::utils::open(in_path.clone()).unwrap();
    eprintln!("height: {},  width: {}, name: {}", img.height(), img.width(), in_path);

    let base_img = cv_sift::pyramid::generate_base_image(&img, 1.6, 0.5)?;

    assert_eq!(img.height() * 2, base_img.height());
    assert_eq!(img.width() * 2, base_img.width());

    eprintln!("height: {},  width: {}, name: {}", base_img.height(), base_img.width(), "base_img");

    Ok(())
}
