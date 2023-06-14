use akaze::Akaze;
use criterion::{criterion_group, criterion_main, Criterion};

fn load_image() -> akaze::image::GrayFloatImage {
    akaze::image::GrayFloatImage::from_dynamic(&image::open("../res/0000000000.png").unwrap())
}

fn extract(c: &mut Criterion) {
    let image = load_image();
    let akaze = Akaze::sparse();
    c.bench_function("extract", |b| {
        b.iter(|| akaze.extract_from_gray_float_image(&image))
    });
}

criterion_group!(
    name = akaze;
    config = Criterion::default().sample_size(10);
    targets = extract
);

fn bench_horizontal_filter(c: &mut Criterion) {
    let image = load_image();
    let small_kernel = akaze::image::gaussian_kernel(1.0, 7);
    c.bench_function("horizontal_filter_small_kernel", |b| {
        b.iter(|| akaze::image::horizontal_filter(&image.0, &small_kernel))
    });
    let large_kernel = akaze::image::gaussian_kernel(10.0, 71);
    c.bench_function("horizontal_filter_large_kernel", |b| {
        b.iter(|| akaze::image::horizontal_filter(&image.0, &large_kernel))
    });
}

fn bench_vertical_filter(c: &mut Criterion) {
    let image = load_image();
    let small_kernel = akaze::image::gaussian_kernel(1.0, 7);
    c.bench_function("vertical_filter_small_kernel", |b| {
        b.iter(|| akaze::image::vertical_filter(&image.0, &small_kernel))
    });
    let large_kernel = akaze::image::gaussian_kernel(10.0, 71);
    c.bench_function("vertical_filter_large_kernel", |b| {
        b.iter(|| akaze::image::vertical_filter(&image.0, &large_kernel))
    });
}

criterion_group!(
    name = akaze_image;
    config = Criterion::default().sample_size(10);
    targets = bench_horizontal_filter, bench_vertical_filter
);

criterion_main!(akaze, akaze_image);
