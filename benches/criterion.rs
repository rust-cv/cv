use akaze::Akaze;
use bitarray::BitArray;
use criterion::{criterion_group, criterion_main, Criterion};
use std::path::Path;

fn image_to_kps(path: impl AsRef<Path>) -> (Vec<akaze::KeyPoint>, Vec<BitArray<64>>) {
    Akaze::sparse().extract_path(path).unwrap()
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
