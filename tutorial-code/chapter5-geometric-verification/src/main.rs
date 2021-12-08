use cv::{
    bitarray::{BitArray, Hamming},
    camera::pinhole::{CameraIntrinsics, CameraIntrinsicsK1Distortion},
    consensus::Arrsac,
    estimate::EightPoint,
    feature::akaze::Akaze,
    image::{
        image::{self, DynamicImage, GenericImageView, Rgba, RgbaImage},
        imageproc::drawing,
    },
    knn::{Knn, LinearKnn},
    nalgebra::{Point2, Vector2},
    sample_consensus::Consensus,
    CameraModel, FeatureMatch, KeyPoint, Pose,
};
use imageproc::pixelops;
use itertools::Itertools;
use palette::{FromColor, Hsv, RgbHue, Srgb};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn main() {
    // Load the image.
    let src_image_a = image::open("res/0000000000.png").expect("failed to open image file");
    let src_image_b = image::open("res/0000000014.png").expect("failed to open image file");

    // Create an instance of `Akaze` with the default settings.
    let akaze = Akaze::default();

    // Extract the features from the image using akaze.
    let (key_points_a, descriptors_a) = akaze.extract(&src_image_a);
    let (key_points_b, descriptors_b) = akaze.extract(&src_image_b);
    let matches = symmetric_matching(&descriptors_a, &descriptors_b);

    // The camera calibration data for 2011_09_26_drive_0035 camera 0 of the Kitti dataset.
    let camera = CameraIntrinsicsK1Distortion::new(
        CameraIntrinsics::identity()
            .focals(Vector2::new(9.842439e+02, 9.808141e+02))
            .principal_point(Point2::new(6.900000e+02, 2.331966e+02)),
        -3.728755e-01,
    );

    // Create an instance of ARRSAC, the consensus algorithm that will find
    // a model that best fits the data. We need to pass in an RNG with good statistical properties
    // for the random sampling process, and xoshiro256++ is an excellent PRNG for this purpose.
    // It is prefered for this example to use a PRNG so we get the same result every time.
    // Note that the inlier threshold is set to 1e-7. This is specific to the dataset.
    let mut consensus = Arrsac::new(1e-7, Xoshiro256PlusPlus::seed_from_u64(0));

    // Create the estimator. In this case it is the well-known Eight-Point algorithm.
    let estimator = EightPoint::new();

    // Take all of the original matches and use the camera model to compute the bearing
    // of each keypoint. The bearing is the direction that the light came from in the camera's
    // reference frame.
    let matches = matches
        .iter()
        .map(|&[a, b]| {
            FeatureMatch(
                camera.calibrate(key_points_a[a]),
                camera.calibrate(key_points_b[b]),
            )
        })
        .collect_vec();

    // Run the consensus process. This will use the estimator to estimate the pose of the camera
    // from random data repeatedly. It does this in an intelligent way to maximize the number of
    // inliers to the model. For convenience, we use .expect() since we expect to get a pose back,
    // but this should not be used in real code.
    let (pose, inliers) = consensus
        .model_inliers(&estimator, matches.iter().copied())
        .expect("we expect to get a pose");

    // Print out the direction the camera moved.
    // Note that the translation of the pose is in the final camera's reference frame
    // and describes the direction that the point cloud (the world) moves to become that reference
    // frame. Therefore, the negation of the translation of the isometry is the actual
    // translation of the camera.
    let translation = -pose.isometry().translation.vector;
    println!("camera moved forward: {}", translation.z);
    println!("camera moved right: {}", translation.x);
    println!("camera moved down: {}", translation.y);

    // Only keep the inlier matches.
    let matches = inliers.iter().map(|&inlier| matches[inlier]).collect_vec();

    // Make a canvas with the `imageproc::drawing` module.
    // We use the blend mode so that we can draw with translucency on the image.
    // We convert the image to rgba8 during this process.
    let canvas_width = src_image_a.dimensions().0 + src_image_b.dimensions().0;
    let canvas_height = std::cmp::max(src_image_a.dimensions().1, src_image_b.dimensions().1);
    let rgba_image_a = src_image_a.to_rgba8();
    let rgba_image_b = src_image_b.to_rgba8();
    let mut canvas = RgbaImage::from_pixel(canvas_width, canvas_height, Rgba([0, 0, 0, 255]));

    // Create closure to render an image at an x offset in a canvas.
    let mut render_image_onto_canvas_x_offset = |image: &RgbaImage, x_offset: u32| {
        let (width, height) = image.dimensions();
        for (x, y) in (0..width).cartesian_product(0..height) {
            canvas.put_pixel(x + x_offset, y, *image.get_pixel(x, y));
        }
    };
    // Render image a in the top left.
    render_image_onto_canvas_x_offset(&rgba_image_a, 0);
    // Render image b just to the right of image a (in the top right).
    render_image_onto_canvas_x_offset(&rgba_image_b, rgba_image_a.dimensions().0);

    // Draw a translucent line for every match.
    for (ix, &FeatureMatch(a_bearing, b_bearing)) in matches.iter().enumerate() {
        // Compute a color by rotating through a color wheel on only the most saturated colors.
        let ix = ix as f64;
        let hsv = Hsv::new(RgbHue::from_radians(ix * 0.1), 1.0, 1.0);
        let rgb = Srgb::from_color(hsv);

        // Draw the line between the keypoints in the two images.
        let point_to_i32_tup =
            |point: KeyPoint, off: u32| (point.x as i32 + off as i32, point.y as i32);
        drawing::draw_antialiased_line_segment_mut(
            &mut canvas,
            point_to_i32_tup(camera.uncalibrate(a_bearing).unwrap(), 0),
            point_to_i32_tup(
                camera.uncalibrate(b_bearing).unwrap(),
                rgba_image_a.dimensions().0,
            ),
            Rgba([
                (rgb.red * 255.0) as u8,
                (rgb.green * 255.0) as u8,
                (rgb.blue * 255.0) as u8,
                255,
            ]),
            pixelops::interpolate,
        );
    }

    // Get the resulting image.
    let out_image = DynamicImage::ImageRgba8(canvas);

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

/// This function performs non-symmetric matching from a to b.
fn matching(a_descriptors: &[BitArray<64>], b_descriptors: &[BitArray<64>]) -> Vec<Option<usize>> {
    let knn_b = LinearKnn {
        metric: Hamming,
        iter: b_descriptors.iter(),
    };
    (0..a_descriptors.len())
        .map(|a_feature| {
            let knn = knn_b.knn(&a_descriptors[a_feature], 2);
            if knn[0].distance + 24 < knn[1].distance {
                Some(knn[0].index)
            } else {
                None
            }
        })
        .collect()
}

/// This function performs symmetric matching between `a` and `b`.
///
/// Symmetric matching requires a feature in `b` to be the best match for a feature in `a`
/// and for the same feature in `a` to be the best match for the same feature in `b`.
/// The feature that a feature matches to in one direction might not be reciprocated.
/// Consider a 1d line. Three features are in a line `X`, `Y`, and `Z` like `X---Y-Z`.
/// `Y` is closer to `Z` than to `X`. The closest match to `X` is `Y`, but the closest
/// match to `Y` is `Z`. Therefore `X` and `Y` do not match symmetrically. However,
/// `Y` and `Z` do form a symmetric match, because the closest point to `Y` is `Z`
/// and the closest point to `Z` is `Y`.
///
/// Symmetric matching is very important for our purposes and gives stronger matches.
fn symmetric_matching(a: &[BitArray<64>], b: &[BitArray<64>]) -> Vec<[usize; 2]> {
    // The best match for each feature in frame a to frame b's features.
    let forward_matches = matching(a, b);
    // The best match for each feature in frame b to frame a's features.
    let reverse_matches = matching(b, a);
    forward_matches
        .into_iter()
        .enumerate()
        .filter_map(move |(aix, bix)| {
            // First we only proceed if there was a sufficient bix match.
            // Filter out matches which are not symmetric.
            // Symmetric is defined as the best and sufficient match of a being b,
            // and likewise the best and sufficient match of b being a.
            bix.map(|bix| [aix, bix])
                .filter(|&[aix, bix]| reverse_matches[bix] == Some(aix))
        })
        .collect()
}
