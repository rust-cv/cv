use argmin::core::{ArgminKV, ArgminOp, Error, Executor, IterState, Observe, ObserverMode};
use cv::nalgebra::{Point3, Unit, Vector3, Vector6};
use cv::{
    camera::pinhole::{CameraIntrinsicsK1Distortion, EssentialMatrix, NormalizedKeyPoint},
    feature::akaze,
    knn::hnsw::{Searcher, HNSW},
    optimize::lm::{LevenbergMarquardt, TerminationReason},
    Bearing, BitArray, CameraModel, CameraPoint, CameraToCamera, Consensus, Estimator,
    FeatureMatch, FeatureWorldMatch, Pose, Projective, TriangulatorObservances,
    TriangulatorRelative, WorldPoint, WorldToCamera,
};
use cv_optimize::{
    many_view_nelder_mead, two_view_nelder_mead, ManyViewConstraint, ManyViewOptimizer,
    TwoViewConstraint, TwoViewOptimizer,
};
use image::DynamicImage;
use log::*;
use maplit::hashmap;
use ndarray::{array, Array2};
use rand::{seq::SliceRandom, Rng};
use slab::Slab;
use space::Neighbor;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

struct OptimizationObserver;

impl<T: ArgminOp> Observe<T> for OptimizationObserver {
    fn observe_iter(&mut self, state: &IterState<T>, _kv: &ArgminKV) -> Result<(), Error> {
        info!(
            "on iteration {} out of {} with total evaluations {} and current cost {}",
            state.iter, state.max_iters, state.cost_func_count, state.cost
        );
        Ok(())
    }
}

type Features = Vec<(NormalizedKeyPoint, BitArray<64>)>;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pair(usize, usize);

impl Pair {
    /// Creates a new pair, cannonicalizing the order of the pair.
    pub fn new(a: usize, b: usize) -> Self {
        Self(std::cmp::min(a, b), std::cmp::max(a, b))
    }
}

struct Frame {
    /// A VSlam::feeds index
    feed: usize,
    /// The keypoints and corresponding descriptors observed on this frame
    features: Features,
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

/// A 3d point in space that has been observed on two or more frames
#[derive(Debug, Clone)]
struct Landmark {
    /// The world coordinate of the landmark.
    point: WorldPoint,
    /// Contains a map from VSlam::views indices to Frame::features indices.
    observances: HashMap<usize, usize>,
}

/// A frame which has been incorporated into a reconstruction.
#[derive(Debug, Clone)]
struct View {
    /// The VSlam::frame index corresponding to this view
    frame: usize,
    /// Pose in the reconstruction of the view
    pose: WorldToCamera,
    /// A map from Frame::features indices to Reconstruction::landmarks indices
    landmarks: HashMap<usize, usize>,
}

/// Frames from a video source
struct Feed {
    /// The camera intrinsics for this feed
    intrinsics: CameraIntrinsicsK1Distortion,
    /// VSlam::frames indices corresponding to each frame of the feed
    frames: Vec<usize>,
    /// The VSlam::reconstructions index currently being tracked
    /// If tracking fails, the reconstruction will be set to None.
    reconstruction: Option<usize>,
}

/// A series of views and points which exist in the same world space
#[derive(Default, Clone)]
struct Reconstruction {
    /// The VSlam::views IDs contained in this reconstruction
    views: Slab<View>,
    /// The landmarks contained in this reconstruction
    landmarks: Slab<Landmark>,
    /// The HNSW to look up all landmarks in the reconstruction
    features: HNSW<BitArray<64>>,
    /// The map from HNSW entries to Reconstruction::landmarks indices
    feature_landmarks: HashMap<usize, usize>,
}

/// Contains the results of a bundle adjust
pub struct BundleAdjust {
    /// The reconstruction the bundle adjust is happening on.
    reconstruction: usize,
    /// Maps VSlam::views IDs to poses
    poses: Vec<(usize, WorldToCamera)>,
}

pub struct VSlam<C, EE, PE, T, R> {
    /// Contains the camera intrinsics for each feed
    feeds: Slab<Feed>,
    /// Contains each one of the ongoing reconstructions
    reconstructions: Slab<Reconstruction>,
    /// Contains all the frames
    frames: Slab<Frame>,
    /// Helper for HNSW.
    searcher: RefCell<Searcher>,
    /// The threshold used for akaze
    akaze_threshold: f64,
    /// The threshold distance below which a match is allowed
    match_threshold: usize,
    /// The number of points to use in optimization of matches
    optimization_points: usize,
    /// The cutoff for the loss function
    loss_cutoff: f64,
    /// The maximum cosine distance permitted in a valid match
    cosine_distance_threshold: f64,
    /// The maximum iterations to optimize two views.
    two_view_patience: usize,
    /// The threshold of mean cosine distance standard deviation that terminates optimization.
    two_view_std_dev_threshold: f64,
    /// The maximum iterations to run two-view optimization and filtering
    two_view_filter_loop_iterations: usize,
    /// The maximum iterations to optimize many views.
    many_view_patience: usize,
    /// The consensus algorithm
    consensus: RefCell<C>,
    /// The essential matrix estimator
    essential_estimator: EE,
    /// The PnP estimator
    pose_estimator: PE,
    /// The triangulation algorithm
    triangulator: T,
    /// The random number generator
    rng: RefCell<R>,
}

impl<C, EE, PE, T, R> VSlam<C, EE, PE, T, R>
where
    C: Consensus<EE, FeatureMatch<NormalizedKeyPoint>>
        + Consensus<PE, FeatureWorldMatch<NormalizedKeyPoint>>,
    EE: Estimator<FeatureMatch<NormalizedKeyPoint>, Model = EssentialMatrix>,
    PE: Estimator<FeatureWorldMatch<NormalizedKeyPoint>, Model = WorldToCamera>,
    T: TriangulatorObservances + Clone,
    R: Rng,
{
    /// Creates an empty vSLAM reconstruction.
    pub fn new(
        consensus: C,
        essential_estimator: EE,
        pose_estimator: PE,
        triangulator: T,
        rng: R,
    ) -> Self {
        Self {
            feeds: Default::default(),
            reconstructions: Default::default(),
            frames: Default::default(),
            searcher: Default::default(),
            akaze_threshold: 0.001,
            match_threshold: 64,
            loss_cutoff: 0.05,
            cosine_distance_threshold: 0.001,
            two_view_patience: 1000,
            two_view_std_dev_threshold: 0.00000001,
            two_view_filter_loop_iterations: 2,
            many_view_patience: 1000,
            optimization_points: 128,
            consensus: RefCell::new(consensus),
            essential_estimator,
            pose_estimator,
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

    /// Set the amount to limit the loss at (lowering reduces the impact of outliers).
    ///
    /// Default: `0.05`
    pub fn loss_cutoff(self, loss_cutoff: f64) -> Self {
        Self {
            loss_cutoff,
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

    /// Set the maximum iterations of two-view optimization.
    ///
    /// Default: `1000`
    pub fn two_view_patience(self, two_view_patience: usize) -> Self {
        Self {
            two_view_patience,
            ..self
        }
    }

    /// Set the maximum iterations of many-view optimization.
    ///
    /// Default: `1000`
    pub fn many_view_patience(self, many_view_patience: usize) -> Self {
        Self {
            many_view_patience,
            ..self
        }
    }

    /// The threshold of mean cosine distance standard deviation that terminates optimization.
    ///
    /// The smaller this value is the more accurate the output will be, but it will take longer to execute.
    ///
    /// Default: `0.00000001`
    pub fn two_view_std_dev_threshold(self, two_view_std_dev_threshold: f64) -> Self {
        Self {
            two_view_std_dev_threshold,
            ..self
        }
    }

    /// Adds a new feed with the given intrinsics.
    pub fn insert_feed(&mut self, intrinsics: CameraIntrinsicsK1Distortion) -> usize {
        self.feeds.insert(Feed {
            intrinsics,
            frames: vec![],
            reconstruction: None,
        })
    }

    /// Add frame.
    ///
    /// This may perform camera tracking and will always extract features.
    ///
    /// Returns a VSlam::reconstructions index if the frame was incorporated in a reconstruction.
    pub fn insert_frame(&mut self, feed: usize, image: &DynamicImage) -> Option<usize> {
        // Extract the features for the frame and add the frame object.
        let next_id = self.frames.insert(Frame {
            feed,
            features: self.kps_descriptors(&self.feeds[feed].intrinsics, image),
        });
        // Add the frame to the feed.
        self.feeds[feed].frames.push(next_id);
        // Get the number of frames this feed has.
        let num_frames = self.feeds[feed].frames.len();

        if let Some(reconstruction) = self.feeds[feed].reconstruction {
            // If the feed has an active reconstruction, try to track the frame.
            if self.try_track(reconstruction, next_id).is_none() {
                // If tracking fails, set the active reconstruction to None.
                self.feeds[feed].reconstruction = None;
            }
        } else if num_frames >= 2 {
            // If there is no active reconstruction, but we have at least two frames, try to initialize the reconstruction
            // using the last two frames.
            let a = self.feeds[feed].frames[num_frames - 2];
            let b = self.feeds[feed].frames[num_frames - 1];
            self.feeds[feed].reconstruction = self.try_init(Pair::new(a, b));
        }
        self.feeds[feed].reconstruction
    }

    fn get_pair(&self, Pair(a, b): Pair) -> Option<(&Frame, &Frame)> {
        Some((self.frames.get(a)?, self.frames.get(b)?))
    }

    /// Attempts to match a frame pair, creating a new reconstruction from a two view pair.
    ///
    /// Returns the VSlam::reconstructions ID if successful.
    fn try_init(&mut self, pair: Pair) -> Option<usize> {
        // Get the two frames.
        let (a, b) = self.get_pair(pair).expect("tried to match an invalid pair");

        // Add the outcome.
        let (pose, landmarks) = self.init_reconstruction(a, b)?;
        Some(self.add_reconstruction(pair, pose, landmarks))
    }

    /// Attempts to track the camera.
    ///
    /// Returns Reconstruction::views index if successful.
    fn try_track(&mut self, reconstruction: usize, frame: usize) -> Option<usize> {
        // Generate the outcome.
        let (pose, landmarks) = self.locate_frame(reconstruction, &self.frames[frame])?;

        // Add the outcome.
        Some(self.incorporate_frame(reconstruction, frame, pose, landmarks))
    }

    /// Incorporate a frame using the precomputed pose and landmark to feature correspondences.
    ///
    /// The `landmarks` vector elements must be in index form FeatureMatch(Reconstruction::landmarks, Frame::features).
    ///
    /// Returns a Reconstruction::views index.
    fn incorporate_frame(
        &mut self,
        reconstruction: usize,
        frame: usize,
        pose: WorldToCamera,
        landmarks: Vec<FeatureMatch<usize>>,
    ) -> usize {
        // Create new view.
        let view = self.reconstructions[reconstruction].views.insert(View {
            frame,
            pose,
            landmarks: landmarks
                .iter()
                .map(|&FeatureMatch(rlmix, fix)| (fix, rlmix))
                .collect(),
        });
        // Add landmark connections where they need to be added.
        for FeatureMatch(lmix, fix) in landmarks {
            // Add the observance to the landmark.
            self.reconstructions[reconstruction].landmarks[lmix]
                .observances
                .insert(view, fix);
            // Add feature to the HNSW.
            let feature_id = self.reconstructions[reconstruction].features.insert(
                self.frames[frame].features[fix].1,
                &mut self.searcher.borrow_mut(),
            );
            self.reconstructions[reconstruction]
                .feature_landmarks
                .insert(feature_id as usize, lmix);
        }
        view
    }

    fn add_reconstruction(
        &mut self,
        pair: Pair,
        pose: CameraToCamera,
        landmarks: Vec<(CameraPoint, FeatureMatch<usize>)>,
    ) -> usize {
        // Create a new empty reconstruction
        let reconstruction = self.reconstructions.insert(Reconstruction::default());
        // Create view A.
        let view_a = self.reconstructions[reconstruction].views.insert(View {
            frame: pair.0,
            pose: WorldToCamera::identity(),
            landmarks: Default::default(),
        });
        // Create view B.
        let view_b = self.reconstructions[reconstruction].views.insert(View {
            frame: pair.1,
            pose: WorldToCamera(pose.0),
            landmarks: Default::default(),
        });
        // Add landmarks.
        for (CameraPoint(point), FeatureMatch(fa, fb)) in landmarks {
            let point = WorldPoint(point);
            // Create the landmark.
            let lmix = self.reconstructions[reconstruction]
                .landmarks
                .insert(Landmark {
                    point,
                    observances: hashmap! {
                        view_a => fa,
                        view_b => fb,
                    },
                });
            // Add the landmark to the two views.
            self.reconstructions[reconstruction].views[view_a]
                .landmarks
                .insert(fa, lmix);
            self.reconstructions[reconstruction].views[view_b]
                .landmarks
                .insert(fb, lmix);
            // Add feature a to the HNSW.
            let feature_a_id = self.reconstructions[reconstruction].features.insert(
                self.frames[pair.0].features[fa].1,
                &mut self.searcher.borrow_mut(),
            );
            self.reconstructions[reconstruction]
                .feature_landmarks
                .insert(feature_a_id as usize, lmix);
            // Add feature b to the HNSW.
            let feature_b_id = self.reconstructions[reconstruction].features.insert(
                self.frames[pair.1].features[fb].1,
                &mut self.searcher.borrow_mut(),
            );
            self.reconstructions[reconstruction]
                .feature_landmarks
                .insert(feature_b_id as usize, lmix);
        }
        reconstruction
    }

    /// Triangulates the point of each match, filtering out matches which fail triangulation or chirality test.
    fn camera_to_camera_match_points<'a>(
        &'a self,
        a: &'a Frame,
        b: &'a Frame,
        pose: CameraToCamera,
        matches: impl Iterator<Item = FeatureMatch<usize>> + 'a,
    ) -> impl Iterator<Item = (CameraPoint, FeatureMatch<usize>)> + 'a {
        matches.filter_map(move |m| {
            let FeatureMatch(a, b) = FeatureMatch(a.keypoint(m.0), b.keypoint(m.1));
            let point_a = self.triangulator.triangulate_relative(pose, a, b)?;
            let point_b = pose.transform(point_a);
            let residual = 1.0 - point_a.bearing().dot(&a.bearing()) + 1.0
                - point_b.bearing().dot(&b.bearing());
            if residual.is_finite()
                && (residual < self.cosine_distance_threshold
                    && point_a.z.is_sign_positive()
                    && point_b.z.is_sign_positive())
            {
                Some((point_a, m))
            } else {
                None
            }
        })
    }

    /// This creates a covisibility between frames `a` and `b` using the essential matrix estimator.
    ///
    /// This method resolves to an undefined scale, and thus is only appropriate for initialization.
    fn init_reconstruction(
        &self,
        a: &Frame,
        b: &Frame,
    ) -> Option<(CameraToCamera, Vec<(CameraPoint, FeatureMatch<usize>)>)> {
        // A helper to convert an index match to a keypoint match given frame a and b.
        let match_ix_kps = |FeatureMatch(aix, bix)| FeatureMatch(a.keypoint(aix), b.keypoint(bix));

        info!(
            "performing brute-force matching between {} and {} features",
            a.features.len(),
            b.features.len()
        );
        // Retrieve the matches which agree with each other from each frame and filter out ones that aren't within the match threshold.
        let matches: Vec<FeatureMatch<usize>> = symmetric_matching(a, b)
            .filter(|&(_, distance)| distance < self.match_threshold)
            .map(|(m, _)| m)
            .collect();

        let original_matches = matches.clone();

        info!("estimate essential on {} matches", matches.len());

        // Estimate the essential matrix and retrieve the inliers
        let (essential, inliers) = self.consensus.borrow_mut().model_inliers(
            &self.essential_estimator,
            matches
                .iter()
                .copied()
                .map(match_ix_kps)
                .collect::<Vec<_>>()
                .iter()
                .copied(),
        )?;
        // Reconstitute only the inlier matches into a matches vector.
        let matches: Vec<FeatureMatch<usize>> = inliers.into_iter().map(|ix| matches[ix]).collect();

        info!("perform chirality test on {}", matches.len());

        // Perform a chirality test to retain only the points in front of both cameras.
        let mut pose = essential
            .pose_solver()
            .solve_unscaled(matches.iter().copied().map(match_ix_kps))?;

        // Initialize the camera points.
        let mut matches: Vec<(CameraPoint, FeatureMatch<usize>)> = self
            .camera_to_camera_match_points(a, b, pose, original_matches.iter().copied())
            .collect();

        for _ in 0..self.two_view_filter_loop_iterations {
            let opti_matches: Vec<FeatureMatch<NormalizedKeyPoint>> = matches
                .choose_multiple(&mut *self.rng.borrow_mut(), self.optimization_points)
                .map(|&(_, m)| m)
                .map(match_ix_kps)
                .collect::<Vec<_>>();

            info!(
                "performing Nelder-Mead optimization on pose using {} matches out of {}",
                opti_matches.len(),
                matches.len()
            );

            let solver = two_view_nelder_mead(pose).sd_tolerance(self.two_view_std_dev_threshold);
            let constraint =
                TwoViewConstraint::new(opti_matches.iter().copied(), self.triangulator.clone())
                    .loss_cutoff(self.loss_cutoff);

            // The initial parameter is empty becasue nelder mead is passed its own initial parameter directly.
            let opti_state = Executor::new(constraint, solver, array![])
                .add_observer(OptimizationObserver, ObserverMode::Always)
                .max_iters(self.two_view_patience as u64)
                .run()
                .expect("two-view optimization failed")
                .state;

            info!(
                "extracted pose with mean capped cosine distance of {}",
                opti_state.best_cost
            );

            pose = Pose::from_se3(Vector6::from_row_slice(
                opti_state
                    .best_param
                    .as_slice()
                    .expect("param was not contiguous array"),
            ));

            // Filter outlier matches based on cosine distance.
            matches = self
                .camera_to_camera_match_points(a, b, pose, original_matches.iter().copied())
                .collect();

            info!("filtering left us with {} matches", matches.len());
        }

        info!(
            "matches remaining after all filtering stages: {}",
            matches.len()
        );

        // Add the new covisibility.
        Some((pose, matches))
    }

    /// Attempts to track the frame in the reconstruction.
    ///
    /// Returns the pose and a vector of indices in the format (Reconstruction::landmarks, Frame::features).
    fn locate_frame(
        &self,
        reconstruction: usize,
        frame: &Frame,
    ) -> Option<(WorldToCamera, Vec<FeatureMatch<usize>>)> {
        let reconstruction = &self.reconstructions[reconstruction];
        info!("find existing landmarks to track camera");
        // Start by trying to match the frame's features to the landmarks in the reconstruction.
        // Get back a bunch of (Reconstruction::landmarks, Frame::features) correspondences.
        let matches: Vec<FeatureMatch<usize>> = frame
            .features
            .iter()
            .enumerate()
            .filter_map(|(ix, &(_, descriptor))| {
                // Find the nearest neighbors.
                let mut neighbors = [Neighbor::invalid(); 1];
                let lm_feature_id = reconstruction
                    .features
                    .nearest(
                        &descriptor,
                        24,
                        &mut self.searcher.borrow_mut(),
                        &mut neighbors,
                    )
                    .first()
                    .cloned()?;
                let distance = reconstruction
                    .features
                    .feature(lm_feature_id.index as u32)
                    .distance(&descriptor);
                // TODO: Perhaps add symmetric filtering and lowes ratio (by landmark, not by feature) filtering here
                if distance < self.match_threshold {
                    Some(FeatureMatch(
                        reconstruction.feature_landmarks[&lm_feature_id.index],
                        ix,
                    ))
                } else {
                    None
                }
            })
            .collect();

        info!("found {} landmark matches", matches.len());

        // Extract the FeatureWorldMatch for each of the features.
        let matches_3d: Vec<FeatureWorldMatch<NormalizedKeyPoint>> = matches
            .iter()
            .map(|&FeatureMatch(lmix, fix)| {
                FeatureWorldMatch(frame.features[fix].0, reconstruction.landmarks[lmix].point)
            })
            .collect();

        info!(
            "estimate the pose of the camera using {} existing landmarks",
            matches_3d.len()
        );

        // Estimate the pose matrix and retrieve the inliers
        let (pose, inliers) = self
            .consensus
            .borrow_mut()
            .model_inliers(&self.pose_estimator, matches_3d.iter().copied())?;

        // Reconstitute only the inlier matches into a matches vector.
        Some((pose, inliers.into_iter().map(|ix| matches[ix]).collect()))
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

    pub fn export_reconstruction(
        &self,
        reconstruction: usize,
        min_observances: usize,
        path: impl AsRef<Path>,
    ) {
        let reconstruction = &self.reconstructions[reconstruction];
        // Output point cloud.
        let points: Vec<Point3<f64>> = reconstruction
            .landmarks
            .iter()
            .filter_map(|(_, lm)| {
                if lm.observances.len() >= min_observances {
                    lm.point.point()
                } else {
                    None
                }
            })
            .collect();
        crate::export::export(std::fs::File::create(path).unwrap(), points);
    }

    /// Optimizes the entire reconstruction.
    ///
    /// Use `num_landmarks` to control the number of landmarks used in optimization.
    pub fn bundle_adjust_highest_observances(
        &mut self,
        reconstruction: usize,
        num_landmarks: usize,
    ) {
        self.apply_bundle_adjust(
            self.compute_bundle_adjust_highest_observances(reconstruction, num_landmarks),
        );
    }

    /// Optimizes the entire reconstruction.
    ///
    /// Use `num_landmarks` to control the number of landmarks used in optimization.
    ///
    /// Returns a series of camera
    fn compute_bundle_adjust_highest_observances(
        &self,
        reconstruction: usize,
        num_landmarks: usize,
    ) -> BundleAdjust {
        let reconstruction_ix = reconstruction;
        let reconstruction = &self.reconstructions[reconstruction];
        // At least one landmark exists or the unwraps below will fail.
        if !reconstruction.landmarks.is_empty() {
            info!(
                "attempting to extract {} landmarks from a total of {}",
                num_landmarks,
                reconstruction.landmarks.len(),
            );

            // First, we want to find the landmarks with the most observances to optimize the reconstruction.
            // Start by putting all the landmark indices into a BTreeMap with the key as their observances and the value the index.
            let mut landmarks_by_observances: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for (observance_num, lmix) in reconstruction
                .landmarks
                .iter()
                .map(|(lmix, lm)| (lm.observances.len(), lmix))
            {
                landmarks_by_observances
                    .entry(observance_num)
                    .or_default()
                    .push(lmix);
            }

            info!(
                "found landmarks with (observances, num) of {:?}",
                landmarks_by_observances
                    .iter()
                    .map(|(ob, v)| (ob, v.len()))
                    .collect::<Vec<_>>()
            );

            // Now the BTreeMap is sorted from smallest number of observances to largest, so take the last indices.
            let mut opti_landmarks: Vec<usize> = vec![];
            for bucket in landmarks_by_observances.values().rev() {
                if opti_landmarks.len() + bucket.len() >= num_landmarks {
                    // Add what we need to randomly (to prevent patterns in data that throw off optimization).
                    opti_landmarks.extend(
                        bucket
                            .choose_multiple(
                                &mut *self.rng.borrow_mut(),
                                num_landmarks - opti_landmarks.len(),
                            )
                            .copied(),
                    );
                    break;
                } else {
                    // Add everything from the bucket.
                    opti_landmarks.extend(bucket.iter().copied());
                }
            }

            // Find all the view IDs corresponding to the landmarks.
            let views: Vec<usize> = opti_landmarks
                .iter()
                .copied()
                .flat_map(|lmix| {
                    reconstruction.landmarks[lmix]
                        .observances
                        .iter()
                        .map(|(&view, _)| view)
                })
                .collect::<BTreeSet<usize>>()
                .into_iter()
                .collect();

            // Form a vector over each landmark that contains a vector of the observances present in each view ID in order above.
            let observances: Vec<Vec<Option<Unit<Vector3<f64>>>>> = opti_landmarks
                .iter()
                .copied()
                .map(|lmix| {
                    let lm = &reconstruction.landmarks[lmix];
                    views
                        .iter()
                        .copied()
                        .map(|view| {
                            lm.observances.get(&view).map(|&feature| {
                                self.frames[reconstruction.views[view].frame].features[feature]
                                    .0
                                    .bearing()
                            })
                        })
                        .collect()
                })
                .collect();

            // Retrieve the view poses
            let poses: Vec<WorldToCamera> = views
                .iter()
                .copied()
                .map(|view| reconstruction.views[view].pose)
                .collect();

            info!(
                "performing Nelder-Mead optimization on {} poses with {} landmarks",
                views.len(),
                opti_landmarks.len(),
            );

            let solver = many_view_nelder_mead(poses).sd_tolerance(self.two_view_std_dev_threshold);
            let constraint = ManyViewConstraint::new(
                observances.iter().map(|v| v.iter().copied()),
                self.triangulator.clone(),
            )
            .loss_cutoff(self.loss_cutoff);

            // The initial parameter is empty becasue nelder mead is passed its own initial parameter directly.
            let opti_state = Executor::new(constraint, solver, Array2::zeros((0, 0)))
                .add_observer(OptimizationObserver, ObserverMode::Always)
                .max_iters(self.many_view_patience as u64)
                .run()
                .expect("many-view optimization failed")
                .state;

            info!(
                "extracted poses with mean capped cosine distance of {}",
                opti_state.best_cost
            );

            let poses: Vec<WorldToCamera> = opti_state
                .best_param
                .outer_iter()
                .map(|arr| {
                    Pose::from_se3(Vector6::from_row_slice(
                        arr.as_slice().expect("param was not contiguous array"),
                    ))
                })
                .collect();

            BundleAdjust {
                reconstruction: reconstruction_ix,
                poses: views.iter().copied().zip(poses).collect(),
            }
        } else {
            BundleAdjust {
                reconstruction: reconstruction_ix,
                poses: vec![],
            }
        }
    }

    pub fn num_reconstructions(&self) -> usize {
        self.reconstructions.len()
    }

    pub fn reconstruction_view_count(&self, reconstruction: usize) -> usize {
        self.reconstructions
            .get(reconstruction)
            .map(|r| r.views.len())
            .unwrap_or(0)
    }

    fn apply_bundle_adjust(&mut self, bundle_adjust: BundleAdjust) {
        let BundleAdjust {
            reconstruction,
            poses,
        } = bundle_adjust;
        for (view, pose) in poses {
            self.reconstructions[reconstruction].views[view].pose = pose;
        }
    }

    pub fn filter_observations(&mut self, reconstruction: usize, threshold: f64) {
        info!("filtering reconstruction observations");
        let landmarks: Vec<usize> = self.reconstructions[reconstruction]
            .landmarks
            .iter()
            .map(|(lmix, _)| lmix)
            .collect();
        // Log the data before filtering.
        let num_observations: usize = self.reconstructions[reconstruction]
            .landmarks
            .iter()
            .map(|(_, lm)| lm.observances.len())
            .sum();
        info!(
            "started with {} landmarks and {} observations",
            landmarks.len(),
            num_observations
        );
        for lmix in landmarks {
            let point = self.reconstructions[reconstruction].landmarks[lmix].point;
            let observances: Vec<(usize, usize)> = self.reconstructions[reconstruction].landmarks
                [lmix]
                .observances
                .iter()
                .map(|(&view, &feature)| (view, feature))
                .collect();

            for (view, feature) in observances {
                let bearing = self.frames[self.reconstructions[reconstruction].views[view].frame]
                    .features[feature]
                    .0
                    .bearing();
                let view_point = self.reconstructions[reconstruction].views[view]
                    .pose
                    .transform(point);
                if 1.0 - bearing.dot(&view_point.bearing()) > threshold {
                    // If the observance has too high of a residual, we must remove it from the landmark and the view.
                    self.reconstructions[reconstruction].landmarks[lmix]
                        .observances
                        .remove(&view);
                    self.reconstructions[reconstruction].views[view]
                        .landmarks
                        .remove(&feature);
                }
            }

            if self.reconstructions[reconstruction].landmarks[lmix]
                .observances
                .len()
                < 2
            {
                // In this case the landmark should be removed, as it no longer has meaning without at least two views.
                self.reconstructions[reconstruction].landmarks.remove(lmix);
            }
        }

        info!("rebuilding HNSW");

        // Rebuild the HNSW of this reconstruction.
        let mut features = HNSW::new();
        let mut feature_landmarks = HashMap::new();
        for (lmix, lm) in self.reconstructions[reconstruction].landmarks.iter() {
            for (&view, &feature) in lm.observances.iter() {
                let fix = features.insert(
                    self.frames[self.reconstructions[reconstruction].views[view].frame].features
                        [feature]
                        .1,
                    &mut self.searcher.borrow_mut(),
                );
                feature_landmarks.insert(fix as usize, lmix);
            }
        }

        // Assign the new HNSW and feature_landmarks map.
        self.reconstructions[reconstruction].features = features;
        self.reconstructions[reconstruction].feature_landmarks = feature_landmarks;

        // Log the data after filtering.
        let num_observations: usize = self.reconstructions[reconstruction]
            .landmarks
            .iter()
            .map(|(_, lm)| lm.observances.len())
            .sum();
        info!(
            "ended with {} landmarks and {} observations",
            self.reconstructions[reconstruction].landmarks.len(),
            num_observations
        );
    }

    pub fn retriangulate_landmarks(&mut self, reconstruction: usize) {
        info!("filtering reconstruction observations");
        let landmarks: Vec<usize> = self.reconstructions[reconstruction]
            .landmarks
            .iter()
            .map(|(lmix, _)| lmix)
            .collect();
        for lmix in landmarks {
            if let Some(point) = self.triangulator.triangulate_observances(
                self.reconstructions[reconstruction].landmarks[lmix]
                    .observances
                    .iter()
                    .map(|(&view, &feature)| {
                        (
                            self.reconstructions[reconstruction].views[view].pose,
                            self.frames[self.reconstructions[reconstruction].views[view].frame]
                                .features[feature]
                                .0,
                        )
                    }),
            ) {
                self.reconstructions[reconstruction].landmarks[lmix].point = point;
            }
        }
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
