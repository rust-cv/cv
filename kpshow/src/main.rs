use akaze::{Akaze, KeyPoint};
use image::{DynamicImage, ImageOutputFormat, Rgba};
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
    /// The output path to write to (autodetects image type from extension).
    ///
    /// If this is not provided, then the output goes to stdout as a PNG.
    #[structopt(short, long, parse(from_os_str))]
    output: Option<PathBuf>,
}

fn main() {
    let opt = Opt::from_args();
    let akaze = Akaze::default();
    let (kps, _) = akaze
        .extract_path(&opt.file)
        .expect("failed to extract features");
    let mut image = drawing::Blend(
        image::open(&opt.file)
            .expect("failed to open image for drawing")
            .into_rgba(),
    );
    for KeyPoint { point: (x, y), .. } in kps {
        drawing::draw_cross_mut(&mut image, Rgba([0, 255, 255, 128]), x as i32, y as i32);
    }
    let stdout = std::io::stdout();
    let image = DynamicImage::ImageRgba8(image.0);
    if let Some(path) = opt.output {
        image.save(path).expect("failed to write image to stdout");
    } else {
        image
            .write_to(&mut stdout.lock(), ImageOutputFormat::Png)
            .expect("failed to write image to stdout");
    }
}
