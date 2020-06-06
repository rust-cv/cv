use cv::nalgebra::Point3;
use cv::{
    camera::pinhole::{CameraIntrinsicsK1Distortion, NormalizedKeyPoint},
    feature::akaze,
    Bearing, BitArray, CameraModel, Consensus, EssentialMatrix, Estimator, FeatureMatch, Pose,
    RelativeCameraPose, TriangulatorObservances, TriangulatorRelative,
};
use cv_optimize::TwoViewOptimizer;
use image::DynamicImage;
use levenberg_marquardt::LevenbergMarquardt;
use log::*;
use rand::{seq::SliceRandom, Rng};
use slab::Slab;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::path::Path;

type Features = Vec<(NormalizedKeyPoint, BitArray<64>)>;

struct Frame {
    /// A VSlam::feeds index
    feed: usize,
    /// The keypoints and corresponding descriptors observed on this frame
    features: Features,
    /// A map from VSlam::landmarks indices to Frame::features indices
    landmarks: HashMap<usize, usize>,
    /// A list of outgoing (first in pair) covisibilities
    covisibilities: HashSet<Pair>,
}

impl Frame {
    fn keypoints(&self) -> impl Iterator<Item = NormalizedKeyPoint> + Clone + '_ {
        self.features.iter().map(|&(kp, _)| kp)
    }

    fn descriptors(&self) -> impl Iterator<Item = BitArray<64>> + Clone + '_ {
        self.features.iter().map(|&(_, d)| d)
    }

    fn keypoint(&self, ix: usize) -> NormalizedKeyPoint {
        self.features[ix].0
    }

    fn descriptor(&self, ix: usize) -> BitArray<64> {
        self.features[ix].1
    }
}

/// The observance of a landmark on a particular frame.
struct Observance {
    /// A VSlam::frames index
    frame: usize,
    /// A Frame::features index
    feature: usize,
}

/// A 3d point in space that has been observed on two or more frames
struct Landmark {
    /// Contains the observances of the landmark on various frames
    observances: Slab<Observance>,
}

struct Feed {
    /// The camera intrinsics for this feed
    intrinsics: CameraIntrinsicsK1Distortion,
    /// VSlam::frames indices corresponding to each frame of the feed
    frames: Vec<usize>,
}

struct Covisibility {
    /// The relative pose between the first to the second frame in the covisibility
    pose: RelativeCameraPose,
    /// The matches in index from the first view to the second
    matches: Vec<FeatureMatch<usize>>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pair(usize, usize);

impl Pair {
    /// Creates a new pair, cannonicalizing the order of the pair.
    pub fn new(a: usize, b: usize) -> Self {
        Self(std::cmp::min(a, b), std::cmp::max(a, b))
    }
}

pub struct VSlam<C, EE, T, R> {
    /// Contains the camera intrinsics for each feed
    feeds: Slab<Feed>,
    /// Contains all the landmarks
    landmarks: Slab<Landmark>,
    /// Contains all the frames
    frames: Slab<Frame>,
    /// Contains a mapping from cannonical pairs to covisibility data
    covisibilities: HashMap<Pair, Option<Covisibility>>,
    /// The threshold used for akaze
    akaze_threshold: f64,
    /// The threshold distance below which a match is allowed
    match_threshold: usize,
    /// The number of points to use in optimization of matches
    optimization_points: usize,
    /// The maximum cosine distance permitted in a valid match
    cosine_distance_threshold: f64,
    /// The consensus algorithm
    consensus: RefCell<C>,
    /// The essential matrix estimator
    essential_estimator: EE,
    /// The triangulation algorithm
    triangulator: T,
    /// The random number generator
    rng: RefCell<R>,
}

impl<C, EE, T, R> VSlam<C, EE, T, R>
where
    EE: Estimator<FeatureMatch<NormalizedKeyPoint>, Model = EssentialMatrix>,
    C: Consensus<EE, FeatureMatch<NormalizedKeyPoint>>,
    T: TriangulatorObservances + Clone,
    R: Rng,
{
    /// Creates an empty vSLAM reconstruction.
    pub fn new(consensus: C, essential_estimator: EE, triangulator: T, rng: R) -> Self {
        Self {
            feeds: Default::default(),
            landmarks: Default::default(),
            frames: Default::default(),
            covisibilities: Default::default(),
            akaze_threshold: 0.001,
            match_threshold: 64,
            cosine_distance_threshold: 0.001,
            optimization_points: 16,
            consensus: RefCell::new(consensus),
            essential_estimator,
            triangulator,
            rng: RefCell::new(rng),
        }
    }

    /// Set the akaze threshold.
    ///
    /// Default: `0.001`
    pub fn akaze_threshold(self, akaze_threshold: f64) -> Self {
        Self {
            akaze_threshold,
            ..self
        }
    }

    /// Set the match threshold.
    ///
    /// Default: `64`
    pub fn match_threshold(self, match_threshold: usize) -> Self {
        Self {
            match_threshold,
            ..self
        }
    }

    /// Set the number of points used for optimization of matching.
    ///
    /// Default: `16`
    pub fn optimization_points(self, optimization_points: usize) -> Self {
        Self {
            optimization_points,
            ..self
        }
    }

    /// Set the maximum cosine distance allowed as a match residual.
    ///
    /// Default: `0.001`
    pub fn cosine_distance_threshold(self, cosine_distance_threshold: f64) -> Self {
        Self {
            cosine_distance_threshold,
            ..self
        }
    }

    /// Adds a new feed with the given intrinsics.
    pub fn insert_feed(&mut self, intrinsics: CameraIntrinsicsK1Distortion) -> usize {
        self.feeds.insert(Feed {
            intrinsics,
            frames: vec![],
        })
    }

    /// Add frame.
    ///
    /// This may perform camera tracking, and thus will take a while.
    ///
    /// Returns a VSlam::frames index.
    pub fn insert_frame(&mut self, feed: usize, image: &DynamicImage) -> usize {
        let next_id = self.frames.insert(Frame {
            feed,
            features: self.kps_descriptors(&self.feeds[feed].intrinsics, image),
            landmarks: Default::default(),
            covisibilities: Default::default(),
        });
        self.feeds[feed].frames.push(next_id);
        let feed_frames = &self.feeds[feed].frames[..];
        if feed_frames.len() >= 2 {
            let a = feed_frames[feed_frames.len() - 2];
            let b = feed_frames[feed_frames.len() - 1];
            self.try_match(Pair::new(a, b));
        }
        next_id
    }

    fn get_pair(&self, Pair(a, b): Pair) -> Option<(&Frame, &Frame)> {
        Some((self.frames.get(a)?, self.frames.get(b)?))
    }

    /// Attempts to match the pair.
    ///
    /// Returns `true` if successful.
    fn try_match(&mut self, pair: Pair) -> bool {
        // First see if the covisibility has already been computed, and if so, don't recompute it.
        if let Some(outcome) = self
            .covisibilities
            .get(&pair)
            .map(|c| c.as_ref().map(|_| pair))
        {
            return outcome.is_some();
        }

        // Get the two frames.
        let (a, b) = self.get_pair(pair).expect("tried to get an invalid pair");

        // Generate the outcome.
        let outcome = self.create_covisibility(a, b);
        let success = outcome.is_some();

        // Add the outcome.
        self.add_covisibility_outcome(pair, outcome);

        success
    }

    fn add_covisibility_outcome(&mut self, pair: Pair, covisibility: Option<Covisibility>) {
        if let Some(covisibility) = covisibility {
            // Add it to the covisibility map.
            self.covisibilities.insert(pair, Some(covisibility));
            // Add the covisibility to the first frame.
            self.frames[pair.0].covisibilities.insert(pair);
        } else {
            // Put a None to indicate the failure.
            self.covisibilities.insert(pair, None);
        }
    }

    fn create_covisibility(&self, a: &Frame, b: &Frame) -> Option<Covisibility> {
        // A helper to convert an index match to a keypoint match given frame a and b.
        let match_ix_kps = |&FeatureMatch(aix, bix)| FeatureMatch(a.keypoint(aix), b.keypoint(bix));

        // Retrieve the matches which agree with each other from each frame and filter out ones that aren't within the match threshold.
        let matches: Vec<FeatureMatch<usize>> = symmetric_matching(a, b)
            .filter(|&(_, distance)| distance < self.match_threshold)
            .map(|(m, _)| m)
            .collect();

        info!("estimate essential");

        // Estimate the essential matrix and retrieve the inliers
        let (essential, inliers) = self.consensus.borrow_mut().model_inliers(
            &self.essential_estimator,
            matches
                .iter()
                .map(match_ix_kps)
                .collect::<Vec<_>>()
                .iter()
                .copied(),
        )?;
        // Reconstitute only the inlier matches into a matches vector.
        let matches: Vec<FeatureMatch<usize>> = inliers.into_iter().map(|ix| matches[ix]).collect();

        info!("perform chirality test");

        // Perform a chirality test to retain only the points in front of both cameras.
        let (pose, inliers) = essential
            .pose_solver()
            .solve_unscaled_inliers(self.triangulator.clone(), matches.iter().map(match_ix_kps))?;
        let matches = inliers
            .iter()
            .map(|&inlier| matches[inlier])
            .collect::<Vec<_>>();

        info!("get optimization matches");

        // Select a random sample of 32 points to use for optimization.
        let opti_matches: Vec<FeatureMatch<NormalizedKeyPoint>> = matches
            .choose_multiple(&mut *self.rng.borrow_mut(), self.optimization_points)
            .map(match_ix_kps)
            .collect();

        info!("performing Levenberg-Marquardt");

        let lm = LevenbergMarquardt::new();
        let pose = lm
            .minimize(TwoViewOptimizer::new(
                opti_matches.iter().copied(),
                pose.0,
                self.triangulator.clone(),
            ))
            .0
            .pose;

        info!("filtering matches using cosine distance");

        // Filter outlier matches based on cosine distance.
        let matches: Vec<FeatureMatch<usize>> = matches
            .iter()
            .filter_map(|m| {
                let FeatureMatch(a, b) = match_ix_kps(m);
                self.triangulator
                    .triangulate_relative(pose, a, b)
                    .map(|point_a| {
                        let point_b = pose.transform(point_a);
                        1.0 - point_a.coords.normalize().dot(&a.bearing()) + 1.0
                            - point_b.coords.normalize().dot(&b.bearing())
                    })
                    .and_then(|residual| {
                        if residual < self.cosine_distance_threshold {
                            Some(*m)
                        } else {
                            None
                        }
                    })
            })
            .collect();

        // Add the new covisibility.
        Some(Covisibility { pose, matches })
    }

    fn kps_descriptors(
        &self,
        intrinsics: &CameraIntrinsicsK1Distortion,
        image: &DynamicImage,
    ) -> Features {
        let (kps, ds) = akaze::Akaze::new(self.akaze_threshold).extract(image);
        kps.into_iter()
            .zip(ds)
            .map(|(kp, d)| (intrinsics.calibrate(kp), d))
            .collect()
    }

    fn covisibility_keypoint_matches<'a>(
        &'a self,
        pair: Pair,
        covisibility: &'a Covisibility,
    ) -> impl Iterator<Item = FeatureMatch<NormalizedKeyPoint>> + 'a {
        covisibility
            .matches
            .iter()
            .map(move |&FeatureMatch(aix, bix)| {
                FeatureMatch(
                    self.frames[pair.0].keypoint(aix),
                    self.frames[pair.1].keypoint(bix),
                )
            })
    }

    pub fn export_covisibility(&self, pair: Pair, path: impl AsRef<Path>) {
        // Get the covisibility object.
        let covisibility = self
            .covisibilities
            .get(&pair)
            .expect("covisibility not found")
            .as_ref()
            .expect("covisibility was found to have been a failure");

        // Output point cloud.
        let points: Vec<Point3<f64>> = self
            .covisibility_keypoint_matches(pair, covisibility)
            .map(|FeatureMatch(a, b)| {
                self.triangulator
                    .triangulate_relative(covisibility.pose, a, b)
                    .unwrap()
                    .0
            })
            .collect();
        crate::export::export(std::fs::File::create(path).unwrap(), points);
    }
}

fn matching(
    a_descriptors: impl Iterator<Item = BitArray<64>>,
    b_descriptors: impl Iterator<Item = BitArray<64>> + Clone,
) -> Vec<(usize, usize)> {
    a_descriptors
        .map(|a| {
            b_descriptors
                .clone()
                .map(|b| a.distance(&b))
                .enumerate()
                .min_by_key(|&(_, d)| d)
                .unwrap()
        })
        .collect::<Vec<_>>()
}

fn symmetric_matching<'a>(
    a: &'a Frame,
    b: &'a Frame,
) -> impl Iterator<Item = (FeatureMatch<usize>, usize)> + 'a {
    // The best match for each feature in frame a to frame b's features.
    let forward_matches = matching(a.descriptors(), b.descriptors());
    // The best match for each feature in frame b to frame a's features.
    let reverse_matches = matching(b.descriptors(), a.descriptors());
    forward_matches
        .into_iter()
        .enumerate()
        .filter_map(move |(aix, (bix, distance))| {
            // Does the feature in b match with this feature too?
            let is_symmetric = reverse_matches[bix].0 == aix;
            if is_symmetric {
                Some((FeatureMatch(aix, bix), distance))
            } else {
                None
            }
        })
}
