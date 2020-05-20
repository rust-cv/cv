use akaze::{Akaze, KeyPoint};
use image::{DynamicImage, ImageOutputFormat, Rgb};
use imageproc::drawing;
use std::path::PathBuf;
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "kpshow",
    about = "A tool to show keypoints from different keypoint detectors."
)]
struct Opt {
    /// The image file to show keypoints on.
    #[structopt(name = "FILE", parse(from_os_str))]
    file: PathBuf,
}

fn main() {
    let opt = Opt::from_args();
    let akaze = Akaze::default();
    let (kps, _) = akaze
        .extract_path(&opt.file)
        .expect("failed to extract features");
    let mut image = image::open(&opt.file)
        .expect("failed to open image for drawing")
        .into_rgb();
    for KeyPoint {
        point: (x, y),
        size,
        ..
    } in kps
    {
        drawing::draw_filled_circle(
            &mut image,
            (x as i32, y as i32),
            size as i32,
            Rgb([0, 255, 255]),
        );
    }
    let stdout = std::io::stdout();
    DynamicImage::ImageRgb8(image)
        .write_to(&mut stdout.lock(), ImageOutputFormat::Png)
        .expect("failed to write image to stdout");
}
