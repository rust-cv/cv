use std::{io::Read, path::PathBuf};
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
#[structopt(name = "imgshow", about = "A tool to display an image")]
struct Opt {
    /// The image file to show.
    ///
    /// If this is not set, stdin will be used to retrieve the image.
    #[structopt(parse(from_os_str))]
    input: Option<PathBuf>,
}

fn main() {
    let opt = Opt::from_args();
    let image = opt
        .input
        .map(|path| image::open(path).expect("failed to open image file"))
        .unwrap_or_else(|| {
            let mut buffer = vec![];
            std::io::stdin()
                .read_to_end(&mut buffer)
                .expect("failed to read stdin to memory");
            image::load_from_memory(&buffer).expect("failed to decode image from stdin")
        });
    imgshow::imgshow(&image)
}
