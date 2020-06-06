use cv::nalgebra::{Unit, Vector3};
use cv::{
    camera::pinhole::{CameraIntrinsicsK1Distortion, NormalizedKeyPoint},
    BitArray, CameraModel, CameraPose, FeatureMatch,
};
use evmap::{ReadHandle, WriteHandle};
use log::*;
use sharded_slab::Slab;
use std::collections::hash_map::RandomState;
use std::sync::{Arc, Mutex};

pub type Features = Vec<(NormalizedKeyPoint, BitArray<64>)>;

pub struct Frame {
    pub feed: usize,
    pub features: Vec<(NormalizedKeyPoint, BitArray<64>)>,
}

/// The appearance of a landmark on a particular frame.
struct Appearance {
    frame: usize,
    feature: usize,
}

/// A 3d point in space that has been observed on two or more frames
struct Landmark {
    position: Vector3<f64>,
    appearances: Slab<Appearance>,
}

struct Feed {
    intrinsics: CameraIntrinsicsK1Distortion,
}

struct Covisibility {
    /// The matches in index from the first view to the second.
    matches: Vec<FeatureMatch<usize>>,
}

/// This can be shared among threads via cloning.
#[derive(Clone)]
pub struct VSlam {
    /// Contains the camera intrinsics for each feed
    feeds: Arc<Slab<Feed>>,
    /// Contains a map from feeds to their frames (in order)
    feed_frames_writer: Arc<Mutex<WriteHandle<usize, usize, (), RandomState>>>,
    feed_frames_reader: ReadHandle<usize, usize, (), RandomState>,
    /// Contains all the landmarks
    landmarks: Arc<Slab<Landmark>>,
    /// Contains all the frames
    frames: Arc<Slab<Frame>>,
}

impl VSlam {
    /// Adds a new feed with the given intrinsics.
    pub fn add_feed(&self, intrinsics: CameraIntrinsicsK1Distortion) -> usize {
        self.feeds.insert(Feed { intrinsics }).unwrap()
    }

    /// Add frame.
    ///
    /// This may perform camera tracking.
    pub fn add_frame(&self, frame: Frame) -> usize {
        unimplemented!()
    }

    /// Creates an empty vSLAM reconstruction.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for VSlam {
    fn default() -> Self {
        let (feed_frames_reader, feed_frames_writer) = evmap::new();
        Self {
            feeds: Arc::new(Slab::new()),
            feed_frames_writer: Arc::new(Mutex::new(feed_frames_writer)),
            feed_frames_reader,
            landmarks: Arc::new(Slab::new()),
            frames: Arc::new(Slab::new()),
        }
    }
}

// fn temporary() {
//     info!("start frame");
//         info!("prev kps: {}", first.0.len());
//         info!("next kps: {}", next.0.len());

//         // Compute best matches from previous frame to next frame and vice versa.
//         let forward_matches = matching(&first.1, &next.1);
//         let reverse_matches = matching(&next.1, &first.1);

//         // Compute the symmetric matches (matches that were the same going forwards and backwards).
//         let matches = forward_matches
//             .iter()
//             .enumerate()
//             .filter_map(|(aix, &(bix, distance))| {
//                 let is_symmetric = reverse_matches[bix].0 == aix;
//                 let in_threshold = distance < opt.match_threshold;
//                 if is_symmetric && in_threshold {
//                     let a = intrinsics.calibrate(prev.0[aix]);
//                     let b = intrinsics.calibrate(next.0[bix]);
//                     Some(FeatureMatch(a, b))
//                 } else {
//                     None
//                 }
//             })
//             .collect::<Vec<_>>();
//         info!("matches: {}", matches.len());

//         // Perform sample consensus.
//         let eight_point = EightPoint::new();
//         let mut arrsac = Arrsac::new(opt.arrsac_threshold, Pcg64::from_seed([1; 32]));
//         let (essential, inliers) = arrsac
//             .model_inliers(&eight_point, matches.iter().copied())
//             .unwrap();
//         let inliers = inliers.iter().map(|&ix| matches[ix]).collect::<Vec<_>>();
//         info!("inliers after sample consensus: {}", inliers.clone().len());
//         let residual_average = inliers
//             .iter()
//             .map(|m| essential.residual(&m).abs())
//             .sum::<f32>()
//             / inliers.len() as f32;
//         info!(
//             "inlier residual average after sample consensus: {}",
//             residual_average
//         );

//         // Perform chirality test to determine which essential matrix is correct.
//         let (pose, inlier_indices) = essential
//             .pose_solver()
//             .solve_unscaled_inliers(
//                 MinimalSquareReprojectionErrorTriangulator::new(),
//                 inliers.clone().into_iter(),
//             )
//             .unwrap();
//         let inliers = inlier_indices
//             .into_iter()
//             .map(|inlier| inliers[inlier])
//             .collect::<Vec<_>>();
//         info!("inliers after chirality test: {}", inliers.len());

//         // Filter outlier matches based on reprojection error.
//         let inliers: Vec<_> = inliers
//             .iter()
//             .cloned()
//             .filter(|m| {
//                 // Get the reprojection error in focal length.
//                 let error_in_focal_length = pinhole::average_pose_reprojection_error(
//                     *pose,
//                     *m,
//                     MinimalSquareReprojectionErrorTriangulator::new(),
//                 )
//                 .unwrap();
//                 // Then convert it to pixels.
//                 let error_in_pixels = error_in_focal_length * net_focal;

//                 debug!("error in pixels: {}", error_in_pixels);

//                 error_in_pixels < opt.reprojection_threshold
//             })
//             .collect();
//         info!(
//             "inliers after reprojection error filtering: {}",
//             inliers.clone().len()
//         );
//         let mean_reprojection_error = inliers
//             .iter()
//             .copied()
//             .map(|m| {
//                 // Get the reprojection error in focal length.
//                 let error_in_focal_length = pinhole::average_pose_reprojection_error(
//                     *pose,
//                     m,
//                     MinimalSquareReprojectionErrorTriangulator::new(),
//                 )
//                 .unwrap();
//                 // Then convert it to pixels.
//                 error_in_focal_length * net_focal
//             })
//             .sum::<f64>()
//             / inliers.len() as f64;
//         info!(
//             "mean reprojection error after reprojection error filtering: {}",
//             mean_reprojection_error
//         );

//         // Output point cloud.
//         if let Some(outpath) = opt.output.as_ref() {
//             let points: Vec<Point3<f64>> = inliers
//                 .iter()
//                 .copied()
//                 .map(|FeatureMatch(a, b)| {
//                     MinimalSquareReprojectionErrorTriangulator::new()
//                         .triangulate_relative(pose.0, a, b)
//                         .unwrap()
//                         .0
//                 })
//                 .collect();
//             export::export(std::fs::File::create(outpath).unwrap(), points);
//         }
//         info!("rotation: {:?}", pose.rotation.angle());
//         info!("end frame");
// }
