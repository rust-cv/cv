use std::{fs, path::Path};

use akaze::Akaze;

fn replace_ext(filename: &str, new: &str) -> String {
    let stemmed = Path::new(filename).file_stem().unwrap().to_str().unwrap();
    format!("{stemmed}{new}")
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    for path in &args[1..] {
        let kps = Akaze::default().extract_path(path).unwrap();
        let mut kp_file = fs::File::create(replace_ext(path, "_kps.csv")).unwrap();
        let mut desc_file = fs::File::create(replace_ext(path, "_descs.txt")).unwrap();
        for (kp, descriptor) in kps.0.iter().zip(kps.1.iter()) {
            std::io::Write::write_all(
                &mut kp_file,
                format!(
                    "{}, {}, {}, {}, {}, {}\n",
                    kp.point.0, kp.point.1, kp.angle, kp.size, kp.octave, kp.class_id
                )
                .as_bytes(),
            )
            .unwrap();
            std::io::Write::write_all(
                &mut desc_file,
                format!("{}\n", descriptor.map(|x| format!("{x:08b}")).join("_")).as_bytes(),
            )
            .unwrap();
        }
    }
}
