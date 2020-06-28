use cv::nalgebra::{Point3, Unit, Vector3};
use cv::{
    camera::pinhole::{CameraIntrinsicsK1Distortion, EssentialMatrix, NormalizedKeyPoint},
    feature::akaze,
    knn::hnsw::{Searcher, HNSW},
    optimize::lm::{LevenbergMarquardt, TerminationReason},
    Bearing, BitArray, CameraModel, CameraPoint, CameraToCamera, Consensus, Estimator,
    FeatureMatch, FeatureWorldMatch, Pose, Projective, TriangulatorObservances,
    TriangulatorRelative, WorldPoint, WorldToCamera,
};
use cv_optimize::{ManyViewOptimizer, TwoViewOptimizer};
use image::DynamicImage;
use log::*;
use maplit::hashmap;
use rand::{seq::SliceRandom, Rng};
use slab::Slab;
use space::Neighbor;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

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
struct Landmark {
    /// The world coordinate of the landmark.
    point: WorldPoint,
    /// Contains a map from VSlam::views indices to Frame::features indices.
    observances: HashMap<usize, usize>,
}

/// A frame which has been incorporated into a reconstruction.
struct View {
    /// The VSlam::frame index corresponding to this view
    frame: usize,
    /// The VSlam::reconstructions index corresponding to this view
    reconstruction: usize,
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
    views: Slab<usize>,
    /// The VSlam::landmarks IDs contained in this reconstruction
    landmarks: Slab<usize>,
    /// The HNSW to look up all landmarks in the reconstruction
    features: HNSW<BitArray<64>>,
    /// The map from HNSW entries to Reconstruction::landmarks IDs
    feature_landmarks: HashMap<usize, usize>,
}

/// Contains the results of a bundle adjust
pub struct BundleAdjust {
    /// The reconstruction the bundle adjust is happening on.
    reconstruction: usize,
    /// Maps VSlam::views IDs to poses
    poses: Vec<(usize, WorldToCamera)>,
    /// Maps VSlam::landmark IDs to points
    points: Vec<(usize, WorldPoint)>,
}

pub struct VSlam<C, EE, PE, T, R> {
    /// Contains the camera intrinsics for each feed
    feeds: Slab<Feed>,
    /// Contains each one of the ongoing reconstructions
    reconstructions: Slab<Reconstruction>,
    /// Contains all the frames
    frames: Slab<Frame>,
    /// Contains all the views
    views: Slab<View>,
    /// Contains a set of all the VSlam::views ID pairs which have already been evaluated
    matches: HashSet<Pair>,
    /// Contains all the landmarks
    landmarks: Slab<Landmark>,
    /// Helper for HNSW.
    searcher: RefCell<Searcher>,
    /// The threshold used for akaze
    akaze_threshold: f64,
    /// The threshold distance below which a match is allowed
    match_threshold: usize,
    /// The number of points to use in optimization of matches
    optimization_points: usize,
    /// The number of times we perform LM and filtering in a loop
    levenberg_marquardt_filter_iterations: usize,
    /// The amount to soften the loss function by
    loss_softener: f64,
    /// The maximum cosine distance permitted in a valid match
    cosine_distance_threshold: f64,
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
            views: Default::default(),
            matches: Default::default(),
            landmarks: Default::default(),
            searcher: Default::default(),
            akaze_threshold: 0.001,
            match_threshold: 64,
            levenberg_marquardt_filter_iterations: 20,
            loss_softener: 0.0002,
            cosine_distance_threshold: 0.001,
            optimization_points: 16,
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

    /// Set the number of iterations to run levenberg marquart and perform filtering.
    ///
    /// Default: `20`
    pub fn levenberg_marquardt_filter_iterations(
        self,
        levenberg_marquardt_filter_iterations: usize,
    ) -> Self {
        Self {
            levenberg_marquardt_filter_iterations,
            ..self
        }
    }

    /// Set the amount to soften the loss by (lowering reduces the impact of outliers).
    ///
    /// Default: `0.0002`
    pub fn loss_softener(self, loss_softener: f64) -> Self {
        Self {
            loss_softener,
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
            reconstruction: None,
        })
    }

    /// Add frame.
    ///
    /// This may perform camera tracking and will always extract features.
    ///
    /// Returns a VSlam::frames index.
    pub fn insert_frame(&mut self, feed: usize, image: &DynamicImage) {
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
    /// Returns VSlam::views index if successful.
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
    /// Returns a VSlam::view index.
    fn incorporate_frame(
        &mut self,
        reconstruction: usize,
        frame: usize,
        pose: WorldToCamera,
        landmarks: Vec<FeatureMatch<usize>>,
    ) -> usize {
        // Create new view.
        let view = self.views.insert(View {
            frame,
            reconstruction,
            pose,
            landmarks: landmarks
                .iter()
                .map(|&FeatureMatch(rlmix, fix)| (fix, rlmix))
                .collect(),
        });
        // Add landmark connections where they need to be added.
        for FeatureMatch(rlmix, fix) in landmarks {
            // Add the observance to the landmark.
            self.landmarks[self.reconstructions[reconstruction].landmarks[rlmix]]
                .observances
                .insert(view, fix);
            // Add feature to the HNSW.
            let feature_id = self.reconstructions[reconstruction].features.insert(
                self.frames[frame].features[fix].1,
                &mut self.searcher.borrow_mut(),
            );
            self.reconstructions[reconstruction]
                .feature_landmarks
                .insert(feature_id as usize, rlmix);
        }
        // Add view to reconstruction.
        self.reconstructions[reconstruction].views.insert(view);
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
        let view_a = self.views.insert(View {
            frame: pair.0,
            reconstruction,
            pose: WorldToCamera::identity(),
            landmarks: Default::default(),
        });
        // Create view B.
        let view_b = self.views.insert(View {
            frame: pair.1,
            reconstruction,
            pose: WorldToCamera(pose.0),
            landmarks: Default::default(),
        });
        // Add landmarks.
        for (CameraPoint(point), FeatureMatch(fa, fb)) in landmarks {
            let point = WorldPoint(point);
            // Create the landmark.
            let lm = self.landmarks.insert(Landmark {
                point,
                observances: hashmap! {
                    view_a => fa,
                    view_b => fb,
                },
            });
            // Add the landmark to the reconstruction.
            let reconstruction_lm = self.reconstructions[reconstruction].landmarks.insert(lm);
            // Add the landmark to the two views.
            self.views[view_a].landmarks.insert(fa, reconstruction_lm);
            self.views[view_b].landmarks.insert(fb, reconstruction_lm);
            // Add feature a to the HNSW.
            let feature_a_id = self.reconstructions[reconstruction].features.insert(
                self.frames[pair.0].features[fa].1,
                &mut self.searcher.borrow_mut(),
            );
            self.reconstructions[reconstruction]
                .feature_landmarks
                .insert(feature_a_id as usize, reconstruction_lm);
            // Add feature b to the HNSW.
            let feature_b_id = self.reconstructions[reconstruction].features.insert(
                self.frames[pair.1].features[fb].1,
                &mut self.searcher.borrow_mut(),
            );
            self.reconstructions[reconstruction]
                .feature_landmarks
                .insert(feature_b_id as usize, reconstruction_lm);
        }
        // Add views to reconstruction.
        self.reconstructions[reconstruction].views.insert(view_a);
        self.reconstructions[reconstruction].views.insert(view_b);
        // Mark the view pair as matched.
        self.matches.insert(pair);
        reconstruction
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
        let mut matches: Vec<(CameraPoint, FeatureMatch<usize>)> = original_matches
            .iter()
            .copied()
            .filter_map(|m| {
                let FeatureMatch(a, b) = match_ix_kps(m);
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
            .collect();

        let opti_matches: Vec<FeatureMatch<NormalizedKeyPoint>> = matches
            .choose_multiple(&mut *self.rng.borrow_mut(), self.optimization_points)
            .map(|&(_, m)| m)
            .map(match_ix_kps)
            .collect::<Vec<_>>();

        info!(
            "performing Levenberg-Marquardt on {} matches out of {}",
            opti_matches.len(),
            matches.len()
        );

        let lm = LevenbergMarquardt::new();
        let (tvo, termination) = lm.minimize(
            TwoViewOptimizer::new(
                opti_matches.iter().copied(),
                pose,
                self.triangulator.clone(),
            )
            .loss_softener(self.loss_softener),
        );
        pose = tvo.pose;

        info!("Levenberg-Marquardt: {:?}", termination);

        // Filter outlier matches based on cosine distance.
        matches = original_matches
            .iter()
            .copied()
            .filter_map(|m| {
                let FeatureMatch(a, b) = match_ix_kps(m);
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
            .collect();

        info!("filtering left us with {} matches", matches.len());

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

        info!("triangulate existing landmarks to track camera");

        // Extract the FeatureWorldMatch for each of the features.
        let matches_3d: Vec<FeatureWorldMatch<NormalizedKeyPoint>> = matches
            .iter()
            .map(|&FeatureMatch(rlmix, fix)| {
                // Get VSlam::landmarks index from Reconstruction::landmarks index.
                let lmix = reconstruction.landmarks[rlmix];
                FeatureWorldMatch(frame.features[fix].0, self.landmarks[lmix].point)
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
        // Output point cloud.
        let points: Vec<Point3<f64>> = self.reconstructions[reconstruction]
            .landmarks
            .iter()
            .filter_map(|(_, &lmix)| {
                let lm = &self.landmarks[lmix];
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
        let reconstruction_data = &self.reconstructions[reconstruction];
        // At least one landmark exists or the unwraps below will fail.
        if !reconstruction_data.landmarks.is_empty() {
            info!(
                "attempting to extract {} landmarks from a total of {}",
                num_landmarks,
                reconstruction_data.landmarks.len(),
            );

            // First, we want to find the landmarks with the most observances to optimize the reconstruction.
            // Start by putting all the landmark indices into a BTreeMap with the key as their observances and the value the index.
            let mut landmarks_by_observances: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
            for (observance_num, rlmix) in reconstruction_data
                .landmarks
                .iter()
                .map(|(rlmix, &lmix)| (self.landmarks[lmix].observances.len(), rlmix))
            {
                landmarks_by_observances
                    .entry(observance_num)
                    .or_default()
                    .push(rlmix);
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
                            .choose_multiple(&mut *self.rng.borrow_mut(), num_landmarks)
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
                .flat_map(|rlmix| {
                    self.landmarks[reconstruction_data.landmarks[rlmix]]
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
                .map(|rlmix| {
                    let lm = &self.landmarks[reconstruction_data.landmarks[rlmix]];
                    views
                        .iter()
                        .copied()
                        .map(|view| {
                            lm.observances.get(&view).map(|&feature| {
                                self.frames[self.views[view].frame].features[feature]
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
                .map(|view| self.views[view].pose)
                .collect();

            info!(
                "performing Levenberg-Marquardt on best {} landmarks and {} views",
                opti_landmarks.len(),
                views.len(),
            );

            let levenberg_marquardt = LevenbergMarquardt::new();
            let (mvo, termination) = levenberg_marquardt.minimize(ManyViewOptimizer::new(
                poses,
                observances.iter().map(|v| v.iter().copied()),
                self.triangulator.clone(),
            ));
            let poses = mvo.poses;
            let points = mvo.points;

            info!(
                "Levenberg-Marquardt terminated with reason {:?}",
                termination
            );

            BundleAdjust {
                reconstruction,
                poses: views.iter().copied().zip(poses).collect(),
                points: opti_landmarks
                    .iter()
                    .copied()
                    .zip(points)
                    .filter_map(|(rlmix, point)| Some((rlmix, point?)))
                    .collect(),
            }
        } else {
            BundleAdjust {
                reconstruction,
                poses: vec![],
                points: vec![],
            }
        }
    }

    fn apply_bundle_adjust(&mut self, bundle_adjust: BundleAdjust) {
        let BundleAdjust {
            reconstruction,
            poses,
            points,
        } = bundle_adjust;
        for (view, pose) in poses {
            self.views[view].pose = pose;
        }
        for (rlmix, point) in points {
            self.landmarks[self.reconstructions[reconstruction].landmarks[rlmix]].point = point;
        }
    }

    pub fn filter_observations(&mut self, reconstruction: usize) {
        info!("filtering reconstruction observations");
        let landmarks: Vec<(usize, usize)> = self.reconstructions[reconstruction]
            .landmarks
            .iter()
            .map(|(rlmix, &lmix)| (rlmix, lmix))
            .collect();
        // Log the data before filtering.
        let num_observations: usize = self.reconstructions[reconstruction]
            .landmarks
            .iter()
            .map(|(_, &lmix)| lmix)
            .map(|lmix| self.landmarks[lmix].observances.len())
            .sum();
        info!(
            "started with {} landmarks and {} observations",
            landmarks.len(),
            num_observations
        );
        for (rlmix, lmix) in landmarks {
            let point = self.landmarks[lmix].point;
            let observances: Vec<(usize, usize)> = self.landmarks[lmix]
                .observances
                .iter()
                .map(|(&view, &feature)| (view, feature))
                .collect();

            for (view, feature) in observances {
                let bearing = self.frames[self.views[view].frame].features[feature]
                    .0
                    .bearing();
                if 1.0 - bearing.dot(&point.bearing()) > self.cosine_distance_threshold {
                    // If the observance has too high of a residual, we must remove it from the landmark and the view.
                    self.landmarks[lmix].observances.remove(&view);
                    self.views[view].landmarks.remove(&feature);
                }
            }

            if self.landmarks[lmix].observances.len() < 2 {
                // In this case the landmark should be removed, as it no longer has meaning without at least two views.
                self.reconstructions[reconstruction].landmarks.remove(rlmix);
                self.landmarks.remove(lmix);
            }
        }

        // Log the data after filtering.
        let num_observations: usize = self.reconstructions[reconstruction]
            .landmarks
            .iter()
            .map(|(_, &lmix)| lmix)
            .map(|lmix| self.landmarks[lmix].observances.len())
            .sum();
        info!(
            "ended with {} landmarks and {} observations",
            self.reconstructions[reconstruction].landmarks.len(),
            num_observations
        );
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
