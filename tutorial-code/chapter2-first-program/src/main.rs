use cv::image::{
    image::{self, DynamicImage, Rgba},
    imageproc::drawing,
};
use rand::Rng;

fn main() {
    // Load the image.
    let src_image = image::open("res/0000000000.png").expect("failed to open image file");

    // Create an RNG so we can draw random points on the image.
    let mut rng = rand::thread_rng();

    // Make a canvas with the `imageproc::drawing` module.
    // We use the blend mode so that we can draw with translucency on the image.
    // We convert the image to rgba8 during this process.
    let mut image_canvas = drawing::Blend(src_image.to_rgba8());

    // Loop 50 times.
    for _ in 0..5000 {
        // Generate a random pixel coordinate on the image.
        let x = rng.gen_range(0..src_image.width()) as i32;
        let y = rng.gen_range(0..src_image.height()) as i32;
        // Use the `imageproc::drawing` module to render a cross on the image.
        drawing::draw_cross_mut(&mut image_canvas, Rgba([0, 255, 255, 128]), x, y);
    }

    // Get the resulting image.
    let out_image = DynamicImage::ImageRgba8(image_canvas.0);

    // Save the image to a temporary file.
    let image_file_path = tempfile::Builder::new()
        .suffix(".png")
        .tempfile()
        .unwrap()
        .into_temp_path();
    out_image.save(&image_file_path).unwrap();

    // Open the image with the system's default application.
    open::that(&image_file_path).unwrap();
    // Some applications may spawn in the background and take a while to begin opening the image,
    // and it isn't clear if its possible to always detect whether the child process has been closed.
    std::thread::sleep(std::time::Duration::from_secs(5));
}
