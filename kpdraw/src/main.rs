use std::{
    io::{Cursor, Write},
    path::PathBuf,
};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "kpdraw",
    about = "A tool to show keypoints from different keypoint detectors"
)]
struct Opt {
    /// The akaze threshold to use.
    ///
    /// 0.01 will be very sparse and 0.0001 will be very dense.
    #[structopt(short, long, default_value = "0.001")]
    threshold: f64,
    /// The output path to write to (autodetects image type from extension).
    ///
    /// If this is not provided, then the output goes to stdout as a PNG.
    #[structopt(short, long, parse(from_os_str))]
    output: Option<PathBuf>,
    /// The image file to show keypoints on.
    #[structopt(parse(from_os_str))]
    input: PathBuf,
}

fn main() {
    let opt = Opt::from_args();
    let image = image::open(opt.input).expect("failed to open image file");
    let image = kpdraw::render_akaze_keypoints(&image, opt.threshold);
    let stdout = std::io::stdout();
    if let Some(path) = opt.output {
        image.save(path).expect("failed to write image to stdout");
    } else {
        let mut output = Cursor::new(Vec::new());
        image
            .write_to(&mut output, image::ImageFormat::Png)
            .expect("failed to write image to stdout");
        stdout
            .lock()
            .write_all(&output.into_inner())
            .expect("failed to write image to stdout");
    }
}
