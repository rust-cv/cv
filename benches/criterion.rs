use criterion::{criterion_group, criterion_main, Criterion};
use space::{Bits512, Hamming};
use std::path::Path;

type Descriptor = Hamming<Bits512>;

fn image_to_kps(path: impl AsRef<Path>) -> (Vec<akaze::Keypoint>, Vec<Descriptor>) {
    let mut akaze_config = akaze::Config::default();
    akaze_config.detector_threshold = 0.01;
    let (_, kps, akaze_ds) = akaze::extract_features(path, akaze_config);
    let ds = akaze_ds
        .into_iter()
        .map(|akaze::Descriptor { vector }| {
            let mut arr = [0; 512 / 8];
            arr[0..61].copy_from_slice(&vector);
            Hamming(Bits512(arr))
        })
        .collect();
    (kps, ds)
}

fn extract(c: &mut Criterion) {
    c.bench_function("extract", |b| b.iter(|| image_to_kps("res/0000000000.png")));
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = extract
);
criterion_main!(benches);
