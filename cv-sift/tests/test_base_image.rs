use test_case::test_case;


#[test_case("testdata/box.png", "testdata"; "box_")]
fn test_base_image(in_path: &str, out_dir: &str) -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let img = cv_sift::open(in_path.clone()).unwrap();
    eprintln!("height: {},  width: {}, name: {}", img.height(), img.width(), in_path);

    let base_img = cv_sift::base_image(&img, 1.6, 0.5);

    assert_eq!(img.height() * 2, base_img.height());
    assert_eq!(img.width() * 2, base_img.width());

    eprintln!("height: {},  width: {}, name: {}", base_img.height(), base_img.width(), "base_img");

    let mut out_path = std::path::PathBuf::from(out_dir);
    out_path.push("box_base_image.png");
    
    base_img.save(out_path).unwrap();

    Ok(())
}
