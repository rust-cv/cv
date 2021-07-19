mod bicubic;
mod export;
mod settings;

pub use export::*;
pub use settings::*;

use argmin::core::{ArgminKV, ArgminOp, Error, Executor, IterState, Observe, ObserverMode};
use bitarray::{BitArray, Hamming};
use cv_core::nalgebra::{Unit, Vector3, Vector6};
use cv_core::{
    sample_consensus::{Consensus, Estimator},
    Bearing, CameraModel, CameraToCamera, FeatureMatch, FeatureWorldMatch, Pose, Projective,
    TriangulatorObservations, TriangulatorRelative, WorldPoint, WorldToCamera, WorldToWorld,
};
use cv_optimize::{
    many_view_nelder_mead, single_view_nelder_mead, two_view_nelder_mead, ManyViewConstraint,
    SingleViewConstraint, TwoViewConstraint,
};
use cv_pinhole::{CameraIntrinsicsK1Distortion, EssentialMatrix, NormalizedKeyPoint};
use hamming_lsh::HammingHasher;
use hgg::HggLite as Hgg;
use image::DynamicImage;
use itertools::{izip, Itertools};
use log::*;
use maplit::hashmap;
use rand::{seq::SliceRandom, Rng};
use slotmap::{new_key_type, DenseSlotMap};
use space::{KnnInsert, KnnMap};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::Path;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

new_key_type! {
    pub struct FeedKey;
    pub struct FrameKey;
    pub struct ViewKey;
    pub struct LandmarkKey;
    pub struct ReconstructionKey;
}

struct OptimizationObserver;

impl<T: ArgminOp> Observe<T> for OptimizationObserver
where
    T::Param: std::fmt::Debug,
{
    fn observe_iter(&mut self, state: &IterState<T>, _kv: &ArgminKV) -> Result<(), Error> {
        debug!(
            "on iteration {} out of {} with total evaluations {} and current cost {}, params {:?}",
            state.iter, state.max_iters, state.cost_func_count, state.cost, state.param
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Feature {
    pub keypoint: NormalizedKeyPoint,
    pub descriptor: BitArray<64>,
    pub color: [u8; 3],
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Pair(usize, usize);

impl Pair {
    /// Creates a new pair, cannonicalizing the order of the pair.
    pub fn new(a: usize, b: usize) -> Self {
        Self(std::cmp::min(a, b), std::cmp::max(a, b))
    }
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Frame {
    /// A VSlam::feeds index
    pub feed: FeedKey,
    /// The keypoints and corresponding descriptors observed on this frame
    pub features: Vec<Feature>,
    /// The views this frame produced.
    pub view: Option<(ReconstructionKey, ViewKey)>,
    /// LSHes for each bag of features on this frame.
    ///
    /// The two `usize` is a [first, second) range of the features vector that corresponds to the bag.
    pub bags: Vec<(BitArray<32>, usize, usize)>,
}

impl Frame {
    pub fn bag_features(&self, bag: usize) -> impl Iterator<Item = &'_ Feature> + Clone {
        let (_, start, end) = self.bags[bag];
        self.features[start..end].iter()
    }

    pub fn bag_descriptors(&self, bag: usize) -> impl Iterator<Item = &'_ BitArray<64>> + Clone {
        self.bag_features(bag).map(|feature| &feature.descriptor)
    }

    pub fn descriptors(&self) -> impl Iterator<Item = &'_ BitArray<64>> + Clone {
        self.features.iter().map(|feature| &feature.descriptor)
    }

    pub fn keypoint(&self, ix: usize) -> NormalizedKeyPoint {
        self.features[ix].keypoint
    }

    pub fn descriptor(&self, ix: usize) -> &BitArray<64> {
        &self.features[ix].descriptor
    }

    pub fn color(&self, ix: usize) -> [u8; 3] {
        self.features[ix].color
    }

    pub fn bags(&self) -> usize {
        self.bags.len()
    }

    pub fn bag_lsh(&self, bag: usize) -> BitArray<32> {
        self.bags[bag].0
    }

    pub fn bag_start(&self, bag: usize) -> usize {
        self.bags[bag].1
    }
}

/// A 3d point in space that has been observed on two or more frames
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Landmark {
    /// Contains a map from VSlam::views indices to Frame::features indices.
    pub observations: HashMap<ViewKey, usize>,
}

/// A frame which has been incorporated into a reconstruction.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct View {
    /// The VSlam::frame index corresponding to this view
    pub frame: FrameKey,
    /// Pose in the reconstruction of the view
    pub pose: WorldToCamera,
    /// A vector containing the Reconstruction::landmarks indices for each feature in the frame
    pub landmarks: Vec<LandmarkKey>,
}

/// Frames from a video source
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Feed {
    /// The camera intrinsics for this feed
    intrinsics: CameraIntrinsicsK1Distortion,
    /// VSlam::frames indices corresponding to each frame of the feed
    frames: Vec<FrameKey>,
}

/// A series of views and points which exist in the same world space
#[derive(Default)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Reconstruction {
    /// The VSlam::views IDs contained in this reconstruction
    pub views: DenseSlotMap<ViewKey, View>,
    /// The landmarks contained in this reconstruction
    pub landmarks: DenseSlotMap<LandmarkKey, Landmark>,
}

/// Contains the results of a bundle adjust
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct BundleAdjustment {
    /// The reconstruction the bundle adjust is happening on.
    reconstruction: ReconstructionKey,
    /// Maps VSlam::views IDs to poses
    poses: Vec<(ViewKey, WorldToCamera)>,
}

/// The mapping data for VSlam.
#[derive(Default)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct VSlamData {
    /// Contains the camera intrinsics for each feed
    feeds: DenseSlotMap<FeedKey, Feed>,
    /// Contains each one of the ongoing reconstructions
    reconstructions: DenseSlotMap<ReconstructionKey, Reconstruction>,
    /// Contains all the frames
    frames: DenseSlotMap<FrameKey, Frame>,
    /// Contains the LSH hasher.
    hasher: HammingHasher<64, 32>,
    /// The HGG to search descriptors for keypoint `(Reconstruction::view, Frame::features)` instances
    lsh_to_bag: Hgg<Hamming, BitArray<32>, (FrameKey, usize)>,
}

impl VSlamData {
    pub fn feed(&self, feed: FeedKey) -> &Feed {
        &self.feeds[feed]
    }

    pub fn frame(&self, frame: FrameKey) -> &Frame {
        &self.frames[frame]
    }

    pub fn keypoint(&self, frame: FrameKey, feature: usize) -> NormalizedKeyPoint {
        self.frames[frame].keypoint(feature)
    }

    pub fn descriptor(&self, frame: FrameKey, feature: usize) -> &BitArray<64> {
        self.frames[frame].descriptor(feature)
    }

    pub fn color(&self, frame: FrameKey, feature: usize) -> [u8; 3] {
        self.frame(frame).color(feature)
    }

    pub fn reconstructions(&self) -> impl Iterator<Item = ReconstructionKey> + '_ {
        self.reconstructions.keys()
    }

    pub fn reconstruction(&self, reconstruction: ReconstructionKey) -> &Reconstruction {
        &self.reconstructions[reconstruction]
    }

    pub fn view(&self, reconstruction: ReconstructionKey, view: ViewKey) -> &View {
        &self.reconstructions[reconstruction].views[view]
    }

    fn view_mut(&mut self, reconstruction: ReconstructionKey, view: ViewKey) -> &mut View {
        &mut self.reconstructions[reconstruction].views[view]
    }

    pub fn view_frame(&self, reconstruction: ReconstructionKey, view: ViewKey) -> FrameKey {
        self.view(reconstruction, view).frame
    }

    pub fn pose(&self, reconstruction: ReconstructionKey, view: ViewKey) -> WorldToCamera {
        self.view(reconstruction, view).pose
    }

    pub fn observation_landmark(
        &self,
        reconstruction: ReconstructionKey,
        view: ViewKey,
        feature: usize,
    ) -> LandmarkKey {
        self.reconstructions[reconstruction].views[view].landmarks[feature]
    }

    pub fn observation_color(
        &self,
        reconstruction: ReconstructionKey,
        view: ViewKey,
        feature: usize,
    ) -> [u8; 3] {
        self.color(self.view_frame(reconstruction, view), feature)
    }

    pub fn observation_keypoint(
        &self,
        reconstruction: ReconstructionKey,
        view: ViewKey,
        feature: usize,
    ) -> NormalizedKeyPoint {
        self.keypoint(self.view_frame(reconstruction, view), feature)
    }

    pub fn is_observation_good(
        &self,
        reconstruction: ReconstructionKey,
        view: ViewKey,
        feature: usize,
        point: WorldPoint,
        threshold: f64,
    ) -> bool {
        let bearing = self
            .observation_keypoint(reconstruction, view, feature)
            .bearing();
        let view_point = self.reconstructions[reconstruction].views[view]
            .pose
            .transform(point);
        let residual = 1.0 - bearing.dot(&view_point.bearing());
        // If the observation is finite and has a low enough residual, it is good.
        residual.is_finite() && residual < threshold
    }

    pub fn landmark(&self, reconstruction: ReconstructionKey, landmark: LandmarkKey) -> &Landmark {
        &self.reconstructions[reconstruction].landmarks[landmark]
    }

    pub fn landmark_mut(
        &mut self,
        reconstruction: ReconstructionKey,
        landmark: LandmarkKey,
    ) -> &mut Landmark {
        &mut self.reconstructions[reconstruction].landmarks[landmark]
    }

    /// Retrieves the (view, feature) iterator from a landmark.
    pub fn landmark_observations(
        &self,
        reconstruction: ReconstructionKey,
        landmark: LandmarkKey,
    ) -> impl Iterator<Item = (ViewKey, usize)> + Clone + '_ {
        self.landmark(reconstruction, landmark)
            .observations
            .iter()
            .map(|(&view, &feature)| (view, feature))
    }

    /// Retrieves only the robust (view, feature) iterator from a landmark.
    ///
    /// The `threshold` is the maximum cosine distance permitted of an observation.
    pub fn landmark_robust_observations(
        &self,
        reconstruction: ReconstructionKey,
        landmark: LandmarkKey,
        point: WorldPoint,
        threshold: f64,
    ) -> impl Iterator<Item = (ViewKey, usize)> + Clone + '_ {
        self.landmark_observations(reconstruction, landmark)
            .filter(move |&(view, feature)| {
                self.is_observation_good(reconstruction, view, feature, point, threshold)
            })
    }

    /// Add a [`Reconstruction`] from two initial frames and good matches between their features.
    pub fn add_reconstruction(
        &mut self,
        frame_a: FrameKey,
        frame_b: FrameKey,
        pose: CameraToCamera,
        matches: Vec<FeatureMatch<usize>>,
    ) -> (ReconstructionKey, (ViewKey, ViewKey)) {
        // Create a new empty reconstruction.
        let reconstruction = self.reconstructions.insert(Reconstruction::default());
        // Add frame A to new reconstruction using an empty set of landmarks so all features are added as new landmarks.
        let view_a = self.add_view(reconstruction, frame_a, Pose::identity(), |_| None);
        // For all feature matches, create a map from the feature ix .
        let landmarks: HashMap<usize, LandmarkKey> = matches
            .into_iter()
            .map(|FeatureMatch(feature_a, feature_b)| {
                (
                    feature_b,
                    self.observation_landmark(reconstruction, view_a, feature_a),
                )
            })
            .collect();
        // Add frame B to new reconstruction using the extracted landmark, bix pairs.
        let view_b = self.add_view(
            reconstruction,
            frame_b,
            WorldToCamera::from(pose.isometry()),
            |feature| landmarks.get(&feature).copied(),
        );
        (reconstruction, (view_a, view_b))
    }

    /// Adds a new View.
    ///
    /// `existing_landmark` is passed a Frame::features index and returns the associated landmark if it exists.
    pub fn add_view(
        &mut self,
        reconstruction: ReconstructionKey,
        frame: FrameKey,
        pose: WorldToCamera,
        existing_landmark: impl Fn(usize) -> Option<LandmarkKey>,
    ) -> ViewKey {
        let view = self.reconstructions[reconstruction].views.insert(View {
            frame,
            pose,
            landmarks: vec![],
        });
        assert!(
            self.frames[frame].view.is_none(),
            "if you are merging reconstructions, you MUST call VSlamData::incorporate_reconstruction"
        );
        self.frames[frame].view = Some((reconstruction, view));

        // Add all of the view's features to the reconstruction.
        for feature in 0..self.frame(frame).features.len() {
            // Check if the feature is part of an existing landmark.
            let landmark = if let Some(landmark) = existing_landmark(feature) {
                // Add this observation to the observations of this landmark.
                self.landmark_mut(reconstruction, landmark)
                    .observations
                    .insert(view, feature);
                landmark
            } else {
                // Create the landmark.
                self.add_landmark(reconstruction, view, feature)
            };
            // Add the Reconstruction::landmark index to the feature landmarks vector for this view.
            self.view_mut(reconstruction, view).landmarks.push(landmark);
        }
        view
    }

    /// Moves all views from one reconstruction to another and then removes the old reconstruction.
    ///
    /// `landmark_map` must map landmarks in `src_reconstruction` to landmarks in `dest_reconstruction`.
    pub fn incorporate_reconstruction(
        &mut self,
        src_reconstruction: ReconstructionKey,
        dest_reconstruction: ReconstructionKey,
        world_transform: WorldToWorld,
        mut landmark_map: HashMap<LandmarkKey, LandmarkKey>,
    ) {
        let src_views: Vec<ViewKey> = self.reconstructions[src_reconstruction]
            .views
            .keys()
            .collect();
        for src_view in src_views {
            let frame = self.view_frame(src_reconstruction, src_view);

            // Transform the pose from the src reconstruction to the dest reconstruction.
            let pose = (world_transform.isometry()
                * self.view(src_reconstruction, src_view).pose.isometry())
            .into();

            // Create the view.
            let dest_view = self.reconstructions[dest_reconstruction]
                .views
                .insert(View {
                    frame,
                    pose,
                    landmarks: vec![],
                });
            // Update the frame's view to point to the new view.
            self.frames[frame].view = Some((dest_reconstruction, dest_view));

            // Add all of the view's features to the reconstruction.
            for feature in 0..self.frame(frame).features.len() {
                let src_landmark = self.observation_landmark(src_reconstruction, src_view, feature);
                // Check if the source landmark is already mapped to a destination landmark.
                let dest_landmark = if let Some(&dest_landmark) = landmark_map.get(&src_landmark) {
                    // Add this observation to the observations of this landmark.
                    self.landmark_mut(dest_reconstruction, dest_landmark)
                        .observations
                        .insert(dest_view, feature);
                    dest_landmark
                } else {
                    // Create the landmark otherwise.
                    let dest_landmark = self.add_landmark(dest_reconstruction, dest_view, feature);
                    landmark_map.insert(src_landmark, dest_landmark);
                    dest_landmark
                };
                // Add the Reconstruction::landmark index to the feature landmarks vector for this view.
                self.view_mut(dest_reconstruction, dest_view)
                    .landmarks
                    .push(dest_landmark);
            }
        }

        self.reconstructions.remove(src_reconstruction);
    }

    /// Creates a new landmark. You must give the landmark at least one observation, as landmarks
    /// without at least one observation are not permitted.
    fn add_landmark(
        &mut self,
        reconstruction: ReconstructionKey,
        view: ViewKey,
        feature: usize,
    ) -> LandmarkKey {
        self.reconstructions[reconstruction]
            .landmarks
            .insert(Landmark {
                observations: hashmap! {
                    view => feature,
                },
            })
    }

    fn apply_bundle_adjust(&mut self, bundle_adjust: BundleAdjustment) {
        let BundleAdjustment {
            reconstruction,
            poses,
        } = bundle_adjust;
        for (view, pose) in poses {
            self.reconstructions[reconstruction].views[view].pose = pose;
        }
    }

    /// Splits the observation into its own landmark.
    ///
    /// Returns the landmark ID (new or old, as necessary).
    fn split_observation(
        &mut self,
        reconstruction: ReconstructionKey,
        view: ViewKey,
        feature: usize,
    ) -> LandmarkKey {
        // Check if this is the only observation in the landmark.
        let old_landmark = self.observation_landmark(reconstruction, view, feature);
        if self
            .landmark(reconstruction, old_landmark)
            .observations
            .len()
            >= 2
        {
            // Since this wasnt the only observation in the landmark, we can split it.
            // Remove the observation from the old_landmark.
            assert_eq!(
                self.landmark_mut(reconstruction, old_landmark)
                    .observations
                    .remove(&view),
                Some(feature)
            );
            // Create the new landmark.
            let new_landmark = self.reconstructions[reconstruction]
                .landmarks
                .insert(Landmark {
                    observations: hashmap! {
                        view => feature,
                    },
                });
            // Assign the landmark ID to the observation.
            self.view_mut(reconstruction, view).landmarks[feature] = new_landmark;
            new_landmark
        } else {
            old_landmark
        }
    }

    /// Merges two landmarks unconditionally. Returns the new landmark ID.
    fn merge_landmarks(
        &mut self,
        reconstruction: ReconstructionKey,
        landmark_a: LandmarkKey,
        landmark_b: LandmarkKey,
    ) -> LandmarkKey {
        let old_landmark = self.reconstructions[reconstruction]
            .landmarks
            .remove(landmark_b)
            .expect("landmark_b didnt exist");
        for (view, feature) in old_landmark.observations {
            // We must start by updating the landmark in the view for this feature.
            self.view_mut(reconstruction, view).landmarks[feature] = landmark_a;
            // Add the observation to landmark A.
            assert!(self
                .landmark_mut(reconstruction, landmark_a)
                .observations
                .insert(view, feature)
                .is_none());
        }
        landmark_a
    }

    /// Get the most visually similar bags to the bags on a given frame.
    ///
    /// Automatically filters out the same frame so the frame wont match to itself.
    ///
    /// Returns (self_bag, found_view, found_bag) pairs for every reconstruction bag match and
    /// found_frame for each frame match.
    #[allow(clippy::type_complexity)]
    fn find_visually_similar_bags(
        &self,
        frame: FrameKey,
        num: usize,
    ) -> (
        Vec<(ReconstructionKey, Vec<(usize, ViewKey, usize)>)>,
        Vec<FrameKey>,
    ) {
        // Get all of the closest `num` bags to each bag in the current frame.
        let bag_matches = (0..self.frames[frame].bags()).flat_map(|self_bag| {
            self.lsh_to_bag
                .knn_values(&self.frames[frame].bag_lsh(self_bag), num)
                .into_iter()
                .map(move |(_, &(found_frame, found_bag))| (self_bag, found_frame, found_bag))
                .filter(|&(_, found_frame, _)| found_frame != frame)
        });

        // This will map reconstructions to their bag matches.
        let mut reconstruction_bags: HashMap<ReconstructionKey, Vec<(usize, ViewKey, usize)>> =
            HashMap::new();
        // This contains bags which have no reconstruction.
        let mut free_bags: HashSet<FrameKey> = HashSet::new();
        // Sort the matches into their respective reconstruction on the free_bags.
        for (self_bag, found_frame, found_bag) in bag_matches {
            if let Some((reconstruction, found_view)) = self.frames[found_frame].view {
                reconstruction_bags
                    .entry(reconstruction)
                    .or_default()
                    .push((self_bag, found_view, found_bag));
            } else {
                free_bags.insert(found_frame);
            }
        }
        // Reconstitute the reconstruction matches into a vector so we can sort them.
        let mut sorted_reconstruction_bags: Vec<(ReconstructionKey, Vec<(usize, ViewKey, usize)>)> =
            reconstruction_bags.into_iter().collect();
        // Sort the reconstructions such that the one with the most bag matches is first.
        sorted_reconstruction_bags
            .sort_unstable_by_key(|(_, bag_matches)| core::cmp::Reverse(bag_matches.len()));

        (sorted_reconstruction_bags, free_bags.into_iter().collect())
    }

    fn add_frame(&mut self, feed: FeedKey, features: Vec<Feature>) -> FrameKey {
        let horizontal_extrema = features
            .iter()
            .map(|f| f.keypoint.x)
            .minmax_by_key(|&v| float_ord::FloatOrd(v))
            .into_option();
        let vertical_extrema = features
            .iter()
            .map(|f| f.keypoint.y)
            .minmax_by_key(|&v| float_ord::FloatOrd(v))
            .into_option();

        let (features, bags) = if let Some(((lower_x, upper_x), (lower_y, upper_y))) =
            horizontal_extrema.zip(vertical_extrema)
        {
            // TODO: This needs to use clustering, this is bad.
            let horizontal_segments = 16;
            let vertical_segments = 9;
            let horizontal_stride = (upper_x - lower_x) / horizontal_segments as f64;
            let vertical_stride = (upper_y - lower_y) / horizontal_segments as f64;

            // Collect all the hashes into their bags.
            let mut feature_groups: Vec<Vec<Feature>> =
                vec![vec![]; horizontal_segments * vertical_segments];
            let more_than_zero = |f| {
                if f <= 0.0 {
                    0.0
                } else {
                    f
                }
            };
            for feature in features {
                let x = more_than_zero((feature.keypoint.x - lower_x) / horizontal_stride) as usize;
                let x = if x >= horizontal_segments {
                    horizontal_segments - 1
                } else {
                    x
                };
                let y = more_than_zero((feature.keypoint.y - lower_y) / vertical_stride) as usize;
                let y = if y >= vertical_segments {
                    vertical_segments - 1
                } else {
                    y
                };
                feature_groups[y * horizontal_segments + x].push(feature);
            }

            // Create the final feature list and bags from the above feature_groups.
            let mut features = vec![];
            let mut bags = vec![];
            for mut feature_group in feature_groups {
                if !feature_group.is_empty() {
                    let lsh = self
                        .hasher
                        .hash_bag(feature_group.iter().map(|f| &f.descriptor));
                    let start = features.len();
                    features.append(&mut feature_group);
                    let end = features.len();
                    bags.push((lsh, start, end));
                }
            }

            (features, bags)
        } else {
            let lsh = self.hasher.hash_bag(features.iter().map(|f| &f.descriptor));
            let end = features.len();
            (features, vec![(lsh, 0, end)])
        };
        let frame = self.frames.insert(Frame {
            feed,
            features,
            view: None,
            bags,
        });
        for (ix, &(lsh, _, _)) in self.frames[frame].bags.iter().enumerate() {
            self.lsh_to_bag.insert(lsh, (frame, ix));
        }
        self.feeds[feed].frames.push(frame);
        frame
    }
}

pub struct VSlam<C, EE, PE, T, R> {
    /// Mapping data
    pub data: VSlamData,
    /// Settings variables
    pub settings: VSlamSettings,
    /// The consensus algorithm
    pub consensus: RefCell<C>,
    /// The essential matrix estimator
    pub essential_estimator: EE,
    /// The PnP estimator
    pub pose_estimator: PE,
    /// The triangulation algorithm
    pub triangulator: T,
    /// The random number generator
    pub rng: RefCell<R>,
}

impl<C, EE, PE, T, R> VSlam<C, EE, PE, T, R>
where
    C: Consensus<EE, FeatureMatch<NormalizedKeyPoint>>
        + Consensus<PE, FeatureWorldMatch<NormalizedKeyPoint>>,
    EE: Estimator<FeatureMatch<NormalizedKeyPoint>, Model = EssentialMatrix>,
    PE: Estimator<FeatureWorldMatch<NormalizedKeyPoint>, Model = WorldToCamera>,
    T: TriangulatorObservations + Clone,
    R: Rng,
{
    /// Creates an empty vSLAM reconstruction.
    pub fn new(
        data: VSlamData,
        settings: VSlamSettings,
        consensus: C,
        essential_estimator: EE,
        pose_estimator: PE,
        triangulator: T,
        rng: R,
    ) -> Self {
        Self {
            data,
            settings,
            consensus: RefCell::new(consensus),
            essential_estimator,
            pose_estimator,
            triangulator,
            rng: RefCell::new(rng),
        }
    }

    /// Adds a new feed with the given intrinsics.
    pub fn add_feed(&mut self, intrinsics: CameraIntrinsicsK1Distortion) -> FeedKey {
        self.data.feeds.insert(Feed {
            intrinsics,
            frames: vec![],
        })
    }

    /// Add frame.
    ///
    /// This may perform camera tracking and will always extract features.
    ///
    /// Returns a `(Reconstruction, View)` pair if the frame was incorporated in a reconstruction.
    /// Returns the `Frame` in all cases.
    pub fn add_frame(&mut self, feed: FeedKey, image: &DynamicImage) -> FrameKey {
        // Extract the features for the frame and add the frame object.
        let features = self.kps_descriptors(&self.data.feeds[feed].intrinsics, image);
        let frame = self.data.add_frame(feed, features);

        // Find the frames which are most visually similar to this frame.
        let (reconstruction_bags, free_frames) = self
            .data
            .find_visually_similar_bags(frame, self.settings.tracking_bow_matches);

        // Try to localize this new frame with all of the similar frames.
        self.try_localize(frame, reconstruction_bags, free_frames);

        frame
    }

    /// Attempts to match a frame pair, creating a new reconstruction from a two view pair.
    ///
    /// Returns the VSlam::reconstructions ID if successful.
    fn try_init(
        &mut self,
        frame_a: FrameKey,
        frame_b: FrameKey,
    ) -> Option<(ReconstructionKey, (ViewKey, ViewKey))> {
        // Add the outcome.
        let (pose, matches) = self.init_reconstruction(frame_a, frame_b)?;
        Some(
            self.data
                .add_reconstruction(frame_a, frame_b, pose, matches),
        )
    }

    /// Attempts to localize a frame or initalize a reconstruction otherwise.
    ///
    /// Only call this on frames that dont already have a reconstruction.
    ///
    /// Returns the (Reconstruction, (View, View)) with the joined reconstruction and the two views if successful.
    #[allow(clippy::type_complexity)]
    fn try_localize(
        &mut self,
        frame: FrameKey,
        reconstruction_bags: Vec<(ReconstructionKey, Vec<(usize, ViewKey, usize)>)>,
        free_frames: Vec<FrameKey>,
    ) -> Option<(ReconstructionKey, ViewKey)> {
        // Handle all the bag matches with existing reconstructions.
        for (dest_reconstruction, bag_matches) in reconstruction_bags {
            if let Some((src_reconstruction, view)) = self.data.frames[frame].view {
                // The current frame is already in a reconstruction.
                if src_reconstruction != dest_reconstruction {
                    // The frame is present in two separate reconstructions. Try to register them together.
                    self.try_merge_reconstructions(
                        src_reconstruction,
                        view,
                        dest_reconstruction,
                        bag_matches,
                    );
                }
                // Otherwise the frame was already in this reconstruction, so we can skip that.
            } else {
                // The current frame is not already in a reconstruction, so it must be incorporated.
                self.incorporate_frame(dest_reconstruction, frame, bag_matches);
            }
        }

        // Until we create a reconstruction, keep trying.
        for found_frame in free_frames {
            if self.try_init(frame, found_frame).is_some() {
                break;
            }
        }

        self.data.frames[frame].view
    }

    /// Triangulates the point of each match, filtering out matches which fail triangulation or chirality test.
    fn camera_to_camera_match_points<'a>(
        &'a self,
        a: &'a Frame,
        b: &'a Frame,
        pose: CameraToCamera,
        matches: impl Iterator<Item = FeatureMatch<usize>> + 'a,
    ) -> impl Iterator<Item = FeatureMatch<usize>> + 'a {
        matches.filter_map(move |m| {
            let FeatureMatch(a, b) = FeatureMatch(a.keypoint(m.0), b.keypoint(m.1));
            let point_a = self.triangulator.triangulate_relative(pose, a, b)?;
            let point_b = pose.transform(point_a);
            let camera_b_bearing_a = pose.isometry() * a.bearing();
            let camera_b_bearing_b = b.bearing();
            let residual = 1.0 - point_a.bearing().dot(&a.bearing()) + 1.0
                - point_b.bearing().dot(&b.bearing());
            let incidence_cosine_distance = 1.0 - camera_b_bearing_a.dot(&camera_b_bearing_b);
            if residual.is_finite()
                && (residual < self.settings.two_view_cosine_distance_threshold
                    && point_a.z.is_sign_positive()
                    && point_b.z.is_sign_positive()
                    && incidence_cosine_distance > self.settings.incidence_minimum_cosine_distance)
            {
                Some(m)
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
        frame_a: FrameKey,
        frame_b: FrameKey,
    ) -> Option<(CameraToCamera, Vec<FeatureMatch<usize>>)> {
        let a = self.data.frame(frame_a);
        let b = self.data.frame(frame_b);
        // A helper to convert an index match to a keypoint match given frame a and b.
        let match_ix_kps = |FeatureMatch(feature_a, feature_b)| {
            FeatureMatch(a.keypoint(feature_a), b.keypoint(feature_b))
        };

        info!(
            "performing brute-force matching between {} and {} features",
            a.features.len(),
            b.features.len(),
        );
        // Retrieve the matches which agree with each other from each frame and filter out ones that aren't within the match threshold.
        let original_matches: Vec<FeatureMatch<usize>> =
            symmetric_matching(a, b, self.settings.match_better_by).collect();

        info!("estimate essential on {} matches", original_matches.len());

        // Estimate the essential matrix and retrieve the inliers
        let (essential, inliers) = self.consensus.borrow_mut().model_inliers(
            &self.essential_estimator,
            original_matches
                .iter()
                .copied()
                .map(match_ix_kps)
                .collect::<Vec<_>>()
                .iter()
                .copied(),
        )?;
        // Reconstitute only the inlier matches into a matches vector.
        let matches: Vec<FeatureMatch<usize>> =
            inliers.into_iter().map(|ix| original_matches[ix]).collect();

        info!("perform chirality test on {}", matches.len());

        // Perform a chirality test to retain only the points in front of both cameras.
        let mut pose = essential
            .pose_solver()
            .solve_unscaled(matches.iter().copied().map(match_ix_kps))?;

        // Initialize the camera points.
        let mut matches: Vec<FeatureMatch<usize>> = self
            .camera_to_camera_match_points(a, b, pose, original_matches.iter().copied())
            .collect();

        for _ in 0..self.settings.two_view_filter_loop_iterations {
            let opti_matches: Vec<FeatureMatch<NormalizedKeyPoint>> = matches
                .choose_multiple(
                    &mut *self.rng.borrow_mut(),
                    self.settings.optimization_points,
                )
                .copied()
                .map(match_ix_kps)
                .collect::<Vec<_>>();

            info!(
                "performing Nelder-Mead optimization on pose using {} matches out of {}",
                opti_matches.len(),
                matches.len()
            );

            let solver =
                two_view_nelder_mead(pose).sd_tolerance(self.settings.two_view_std_dev_threshold);
            let constraint =
                TwoViewConstraint::new(opti_matches.iter().copied(), self.triangulator.clone())
                    .loss_cutoff(self.settings.loss_cutoff);

            // The initial parameter is empty becasue nelder mead is passed its own initial parameter directly.
            let opti_state = Executor::new(constraint, solver, vec![])
                .add_observer(OptimizationObserver, ObserverMode::Always)
                .max_iters(self.settings.two_view_patience as u64)
                .run()
                .expect("two-view optimization failed")
                .state;

            info!(
                "extracted pose with mean capped cosine distance of {}",
                opti_state.best_cost
            );

            pose = Pose::from_se3(Vector6::from_row_slice(&opti_state.best_param));

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

    /// Attempts to register the frame into the given reconstruction.
    ///
    /// Returns the pose and a map from feature indices to landmarks.
    fn register_frame(
        &mut self,
        reconstruction_key: ReconstructionKey,
        new_frame_key: FrameKey,
        bag_matches: Vec<(usize, ViewKey, usize)>,
    ) -> Option<(WorldToCamera, HashMap<usize, LandmarkKey>)> {
        info!("trying to register frame into existing reconstruction");
        let reconstruction = self.data.reconstruction(reconstruction_key);
        let new_frame = self.data.frame(new_frame_key);

        info!(
            "performing brute-force matching on {} bag pairs",
            bag_matches.len()
        );

        // Retrieve the matches which agree with each other from each frame and filter out ones that aren't within the match threshold.
        let inital_landmark_feature_matches =
            bag_matches
                .into_iter()
                .flat_map(|(new_frame_bag, found_view, found_bag)| {
                    let found_view = &reconstruction.views[found_view];
                    let found_frame = found_view.frame;
                    symmetric_matching_bags(
                        &self.data.frames[found_frame],
                        found_bag,
                        new_frame,
                        new_frame_bag,
                        self.settings.match_better_by,
                    )
                    .map(
                        move |(FeatureMatch(existing_feature, new_feature), distance)| {
                            (
                                found_view.landmarks[existing_feature],
                                new_feature,
                                distance,
                            )
                        },
                    )
                });

        // We need to get only the matches out which do not violate certain rules.
        // Specifically, two separate features cannot match the same landmark.
        // Additionally, two separate landmarks cannot match the same feature.
        // If this happens, all of the instances of this landmark or feature must be removed
        // from the matches entirely.
        // To counter this, we will perform a symmetric match + lowes-like/match_better_by criteria.
        // Only the best matches will be kept, and matches which violate the filter criteria will be removed.
        let mut feature_landmarks: HashMap<usize, Vec<(LandmarkKey, u32)>> = HashMap::new();
        for (landmark, feature, distance) in inital_landmark_feature_matches {
            let landmarks = feature_landmarks.entry(feature).or_default();
            // Don't add duplicate landmarks.
            if landmarks.iter().any(|&(l, _)| l == landmark) {
                continue;
            }
            let pos = landmarks.partition_point(|&(_, d)| d <= distance);
            landmarks.insert(pos, (landmark, distance));
        }
        let landmark_features_filtered = feature_landmarks
            .into_iter()
            .filter(|(_, landmarks)| {
                landmarks.len() == 1
                    || landmarks[0].1 + self.settings.match_better_by < landmarks[1].1
            })
            .map(|(feature, landmarks)| (landmarks[0].0, feature, landmarks[0].1));

        // Next we need to do the same thing for the landmarks.
        let mut landmark_features: HashMap<LandmarkKey, Vec<(usize, u32)>> = HashMap::new();
        for (landmark, feature, distance) in landmark_features_filtered {
            let features = landmark_features.entry(landmark).or_default();
            let pos = features.partition_point(|&(_, d)| d <= distance);
            features.insert(pos, (feature, distance));
        }
        let landmark_features_filtered = landmark_features
            .into_iter()
            .filter(|(_, features)| {
                features.len() == 1 || features[0].1 + self.settings.match_better_by < features[1].1
            })
            .map(|(landmark, features)| (landmark, features[0].0));

        let matches: Vec<(LandmarkKey, usize)> = landmark_features_filtered.collect();

        info!("found {} initial feature matches", matches.len());

        let create_3d_matches = |robust| {
            matches
                .choose_multiple(&mut *self.rng.borrow_mut(), matches.len())
                .filter_map(|&(landmark, feature)| {
                    Some(FeatureWorldMatch(
                        new_frame.features[feature].keypoint,
                        if robust {
                            self.triangulate_landmark_robust(reconstruction_key, landmark)?
                        } else {
                            self.triangulate_landmark(reconstruction_key, landmark)?
                        },
                    ))
                })
                .take(self.settings.track_landmarks)
                .collect()
        };

        info!("retrieving only robust landmarks corresponding to matches");

        // Extract the FeatureWorldMatch for each of the features.
        let matches_3d: Vec<FeatureWorldMatch<NormalizedKeyPoint>> = create_3d_matches(true);

        let matches_3d = if matches_3d.len() < 32 {
            info!("unable to find enough robust landmarks, trying all triangulatable landmarks");
            create_3d_matches(false)
        } else {
            matches_3d
        };

        if matches_3d.len() < self.settings.single_view_minimum_landmarks {
            info!(
                "unable to find at least {} 3d matches to track frame (found {})",
                self.settings.single_view_minimum_landmarks,
                matches_3d.len()
            );
            return None;
        }

        info!(
            "estimate the pose of the camera using {} triangulatable landmarks",
            matches_3d.len()
        );

        // Estimate the pose and retrieve the inliers.
        let pose = self
            .consensus
            .borrow_mut()
            .model(&self.pose_estimator, matches_3d.iter().copied())?;

        // Create solver and constraint for single-view optimizer.
        let solver =
            single_view_nelder_mead(pose).sd_tolerance(self.settings.single_view_std_dev_threshold);
        let constraint =
            SingleViewConstraint::new(matches_3d).loss_cutoff(self.settings.loss_cutoff);

        // The initial parameter is empty becasue nelder mead is passed its own initial parameter directly.
        let opti_state = Executor::new(constraint, solver, vec![])
            .add_observer(OptimizationObserver, ObserverMode::Always)
            .max_iters(self.settings.single_view_patience as u64)
            .run()
            .expect("single-view optimization failed")
            .state;

        info!(
            "extracted single-view pose with mean capped cosine distance of {}",
            opti_state.best_cost
        );

        let pose = Pose::from_se3(Vector6::from_row_slice(&opti_state.best_param));

        // Filter outlier matches and return all others for inclusion.
        let matches: HashMap<usize, LandmarkKey> = matches
            .into_iter()
            .filter(|&(landmark, feature)| {
                let keypoint = new_frame.features[feature].keypoint;
                self.triangulate_landmark_with_appended_observations_and_verify(
                    reconstruction_key,
                    landmark,
                    std::iter::once((pose, keypoint)),
                )
                .is_some()
            })
            .map(|(landmark, feature)| (feature, landmark))
            .collect();

        info!("registered frame with a total of {} matches", matches.len());

        Some((pose, matches))
    }

    /// Attempts to track the frame in the reconstruction.
    ///
    /// Returns the pose and a vector of indices in the format (Reconstruction::landmarks, Frame::features).
    fn incorporate_frame(
        &mut self,
        reconstruction: ReconstructionKey,
        new_frame: FrameKey,
        bag_matches: Vec<(usize, ViewKey, usize)>,
    ) -> Option<ViewKey> {
        let (pose, matches) = self.register_frame(reconstruction, new_frame, bag_matches)?;

        let new_view_key = self
            .data
            .add_view(reconstruction, new_frame, pose, |feature| {
                matches.get(&feature).copied()
            });

        Some(new_view_key)
    }

    /// Attempts to register the given frame in the given source reconstruction with the landmarks in the given
    /// view in the destination reconstruction. It also merges the landmarks between the two reconstructions.
    ///
    /// Returns the resultant (reconstruction, (src_view, dest_view)) pair if it was successful.
    /// The views are now all in the final reconstruction space.
    fn try_merge_reconstructions(
        &mut self,
        src_reconstruction: ReconstructionKey,
        src_view: ViewKey,
        dest_reconstruction: ReconstructionKey,
        dest_bag_matches: Vec<(usize, ViewKey, usize)>,
    ) -> Option<ReconstructionKey> {
        let src_frame = self.data.view_frame(src_reconstruction, src_view);
        // Register the frame in the source reconstruction to the destination reconstruction.
        let (pose, matches) =
            self.register_frame(dest_reconstruction, src_frame, dest_bag_matches)?;

        // Create the transformation from the source to the destination reconstruction.
        let world_transform = WorldToWorld::from_camera_poses(
            self.data.view(src_reconstruction, src_view).pose,
            pose,
        );

        // Create a map from src landmarks to dest landmarks.
        let landmark_to_landmark: HashMap<LandmarkKey, LandmarkKey> = matches
            .iter()
            .map(|(&src_feature, &dest_landmark)| {
                (
                    self.data.view(src_reconstruction, src_view).landmarks[src_feature],
                    dest_landmark,
                )
            })
            .collect();

        self.data.incorporate_reconstruction(
            src_reconstruction,
            dest_reconstruction,
            world_transform,
            landmark_to_landmark,
        );

        Some(dest_reconstruction)
    }

    fn kps_descriptors(
        &self,
        intrinsics: &CameraIntrinsicsK1Distortion,
        image: &DynamicImage,
    ) -> Vec<Feature> {
        let (keypoints, descriptors) =
            akaze::Akaze::new(self.settings.akaze_threshold).extract(image);
        let rbg_image = image.to_rgb8();

        // Use bicubic interpolation to extract colors from the image.
        let colors: Vec<[u8; 3]> = keypoints
            .iter()
            .map(|kp| {
                use image::Rgb;
                let (x, y) = kp.point;
                let Rgb(color) = bicubic::interpolate_bicubic(&rbg_image, x, y, Rgb([0, 0, 0]));
                color
            })
            .collect();

        // Calibrate keypoint and combine into features.
        izip!(
            keypoints.into_iter().map(|kp| intrinsics.calibrate(kp)),
            descriptors,
            colors
        )
        .map(|(keypoint, descriptor, color)| Feature {
            keypoint,
            descriptor,
            color,
        })
        .collect()
    }

    pub fn export_reconstruction(&self, reconstruction: ReconstructionKey, path: impl AsRef<Path>) {
        // Output point cloud.
        let points_and_colors = self
            .data
            .reconstruction(reconstruction)
            .landmarks
            .iter()
            .filter_map(|(landmark, lm_object)| {
                self.triangulate_landmark_robust(reconstruction, landmark)
                    .and_then(Projective::point)
                    .map(|p| {
                        let (&view, &feature) = lm_object.observations.iter().next().unwrap();
                        (
                            p,
                            self.data.observation_color(reconstruction, view, feature),
                        )
                    })
            })
            .collect();
        crate::export::export(std::fs::File::create(path).unwrap(), points_and_colors);
    }

    /// Runs bundle adjustment (camera pose optimization), landmark filtering, and landmark merging.
    pub fn optimize_reconstruction(&mut self, reconstruction: ReconstructionKey) {
        for _ in 0..self.settings.reconstruction_optimization_iterations {
            // If there are three or more views, run global bundle-adjust.
            self.bundle_adjust_reconstruction(reconstruction);
            // Filter observations after running bundle-adjust.
            self.filter_observations(reconstruction);
            // Merge landmarks.
            self.merge_nearby_landmarks(reconstruction);
        }
    }

    /// Optimizes reconstruction camera poses.
    pub fn bundle_adjust_reconstruction(&mut self, reconstruction: ReconstructionKey) {
        self.data
            .apply_bundle_adjust(self.compute_bundle_adjust(reconstruction));
    }

    fn retrieve_top_landmarks(
        &self,
        reconstruction: ReconstructionKey,
        num: usize,
        filter: impl Fn(LandmarkKey) -> bool,
        quality: impl Fn(LandmarkKey) -> usize,
    ) -> Vec<LandmarkKey> {
        info!(
            "attempting to extract {} landmarks from a total of {}",
            num,
            self.data.reconstruction(reconstruction).landmarks.len(),
        );

        // First, we want to find the landmarks with the most observances to optimize the reconstruction.
        // Start by putting all the landmark indices into a BTreeMap with the key as their observances and the value the index.
        let mut landmarks_by_quality: BTreeMap<usize, Vec<LandmarkKey>> = BTreeMap::new();
        for (observations, landmark) in self
            .data
            .reconstruction(reconstruction)
            .landmarks
            .keys()
            .filter(|&landmark| filter(landmark))
            .map(|landmark| (quality(landmark), landmark))
        {
            // Only add landmarks with at least 3 observations.
            landmarks_by_quality
                .entry(observations)
                .or_default()
                .push(landmark);
        }

        info!(
            "found landmarks with (quality, num) of {:?}",
            landmarks_by_quality
                .iter()
                .map(|(ob, v)| (ob, v.len()))
                .collect::<Vec<_>>()
        );

        // Now the BTreeMap is sorted from smallest number of observances to largest, so take the last indices.
        let mut top_landmarks: Vec<LandmarkKey> = vec![];
        for bucket in landmarks_by_quality.values().rev() {
            if top_landmarks.len() + bucket.len() >= num {
                // Add what we need to randomly (to prevent patterns in data that throw off optimization).
                top_landmarks.extend(
                    bucket
                        .choose_multiple(&mut *self.rng.borrow_mut(), num - top_landmarks.len())
                        .copied(),
                );
                break;
            } else {
                // Add everything from the bucket.
                top_landmarks.extend(bucket.iter().copied());
            }
        }
        top_landmarks
    }

    /// Optimizes the entire reconstruction.
    ///
    /// Use `num_landmarks` to control the number of landmarks used in optimization.
    ///
    /// Returns a series of camera
    fn compute_bundle_adjust(&self, reconstruction: ReconstructionKey) -> BundleAdjustment {
        // At least one landmark exists or the unwraps below will fail.
        if !self
            .data
            .reconstruction(reconstruction)
            .landmarks
            .is_empty()
        {
            info!("trying to extract landmarks using robust observations as a quality metric");

            let opti_landmarks = self.retrieve_top_landmarks(
                reconstruction,
                self.settings.many_view_landmarks,
                |landmark| self.is_landmark_robust(reconstruction, landmark),
                |landmark| {
                    self.landmark_robust_observations(reconstruction, landmark)
                        .count()
                },
            );

            let opti_landmarks = if opti_landmarks.len() < 32 {
                info!(
                    "insufficient landmarks ({}), need 32; using non-robust observations as quality metric",
                    opti_landmarks.len()
                );
                let opti_landmarks = self.retrieve_top_landmarks(
                    reconstruction,
                    self.settings.many_view_landmarks,
                    |landmark| {
                        self.triangulate_landmark(reconstruction, landmark)
                            .is_some()
                    },
                    |landmark| {
                        self.data
                            .landmark(reconstruction, landmark)
                            .observations
                            .len()
                    },
                );
                if opti_landmarks.len() < 32 {
                    info!(
                        "insufficient landmarks ({}), need 32; bundle adjust failed",
                        opti_landmarks.len()
                    );
                    return BundleAdjustment {
                        reconstruction,
                        poses: vec![],
                    };
                } else {
                    info!("succeeded with {} landmarks", opti_landmarks.len());
                    opti_landmarks
                }
            } else {
                info!("succeeded with {} landmarks", opti_landmarks.len());
                opti_landmarks
            };

            // Find all the view IDs corresponding to the landmarks.
            let views: Vec<ViewKey> = opti_landmarks
                .iter()
                .copied()
                .flat_map(|landmark| {
                    self.data
                        .landmark(reconstruction, landmark)
                        .observations
                        .iter()
                        .map(|(&view, _)| view)
                })
                .collect::<BTreeSet<ViewKey>>()
                .into_iter()
                .collect();

            // Form a vector over each landmark that contains a vector of the observances present in each view ID in order above.
            let observances: Vec<Vec<Option<Unit<Vector3<f64>>>>> = opti_landmarks
                .iter()
                .copied()
                .map(|landmark| {
                    views
                        .iter()
                        .copied()
                        .map(|view| {
                            self.data
                                .landmark(reconstruction, landmark)
                                .observations
                                .get(&view)
                                .map(|&feature| {
                                    self.data
                                        .observation_keypoint(reconstruction, view, feature)
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
                .map(|view| self.data.pose(reconstruction, view))
                .collect();

            info!(
                "performing Nelder-Mead optimization on {} poses with {} landmarks",
                views.len(),
                opti_landmarks.len(),
            );

            let solver = many_view_nelder_mead(poses)
                .sd_tolerance(self.settings.many_view_std_dev_threshold);
            let constraint = ManyViewConstraint::new(
                observances.iter().map(|v| v.iter().copied()),
                self.triangulator.clone(),
            )
            .loss_cutoff(self.settings.loss_cutoff);

            // The initial parameter is empty becasue nelder mead is passed its own initial parameter directly.
            let opti_state = Executor::new(constraint, solver, vec![])
                .add_observer(OptimizationObserver, ObserverMode::Always)
                .max_iters(self.settings.many_view_patience as u64)
                .run()
                .expect("many-view optimization failed")
                .state;

            info!(
                "extracted poses with mean capped cosine distance of {}",
                opti_state.best_cost
            );

            let poses: Vec<WorldToCamera> = opti_state
                .best_param
                .iter()
                .map(|arr| Pose::from_se3(Vector6::from_row_slice(arr)))
                .collect();

            BundleAdjustment {
                reconstruction,
                poses: views.iter().copied().zip(poses).collect(),
            }
        } else {
            warn!(
                "tried to bundle adjust reconstruction with no landmarks, which should not exist"
            );
            BundleAdjustment {
                reconstruction,
                poses: vec![],
            }
        }
    }

    /// Splits all observations in the landmark into their own separate landmarks.
    fn split_landmark(&mut self, reconstruction: ReconstructionKey, landmark: LandmarkKey) {
        let observations: Vec<(ViewKey, usize)> = self
            .data
            .landmark_observations(reconstruction, landmark)
            .collect();
        // Don't split the first observation off, as it can stay as this landmark.
        for &(view, feature) in &observations[1..] {
            self.data.split_observation(reconstruction, view, feature);
        }
    }

    pub fn filter_observations(&mut self, reconstruction: ReconstructionKey) {
        info!("filtering reconstruction observations");
        let landmarks: Vec<LandmarkKey> = self
            .data
            .reconstruction(reconstruction)
            .landmarks
            .iter()
            .map(|(lmix, _)| lmix)
            .collect();

        // Log the data before filtering.
        let num_triangulatable_landmarks: usize = self
            .data
            .reconstruction(reconstruction)
            .landmarks
            .iter()
            .filter(|&(_, lm)| lm.observations.len() >= 2)
            .count();
        info!(
            "started with {} triangulatable landmarks",
            num_triangulatable_landmarks,
        );

        for landmark in landmarks {
            if let Some(point) = self.triangulate_landmark(reconstruction, landmark) {
                let observations: Vec<(ViewKey, usize)> = self
                    .data
                    .landmark_observations(reconstruction, landmark)
                    .collect();

                for (view, feature) in observations {
                    if !self.data.is_observation_good(
                        reconstruction,
                        view,
                        feature,
                        point,
                        self.settings.cosine_distance_threshold,
                    ) {
                        // If the observation is bad, we must remove it from the landmark and the view.
                        self.data.split_observation(reconstruction, view, feature);
                    }
                }
            } else {
                self.split_landmark(reconstruction, landmark);
            }
        }

        // Log the data after filtering.
        let num_triangulatable_landmarks: usize = self
            .data
            .reconstruction(reconstruction)
            .landmarks
            .iter()
            .filter(|&(_, lm)| lm.observations.len() >= 2)
            .count();
        info!(
            "ended with {} triangulatable landmarks",
            num_triangulatable_landmarks,
        );
    }

    /// Filters landmarks that arent robust.
    ///
    /// It is recommended not to perform this stage normally, as only robust observations are used in the
    /// optimization process. It can be beneficial to have landmarks that are either not triangulatable
    /// or are infinitely far away from the camera (such as stars). Additionally, having weaker obervations
    /// allows the potential for a landmark to become robust in the future. Use this if you really want to strip
    /// out useful data from the reconstruction and only leave the most robust data used for optimization purposes.
    /// This may be useful to do if the amount of data is too large and needs to be trimmed down to only the useful data
    /// for image registration.
    pub fn filter_non_robust_landmarks(&mut self, reconstruction: ReconstructionKey) {
        info!("filtering non-robust landmarks");
        let landmarks: Vec<LandmarkKey> = self
            .data
            .reconstruction(reconstruction)
            .landmarks
            .keys()
            .collect();

        // Log the data before filtering.
        let num_3d_landmarks: usize = self
            .data
            .reconstruction(reconstruction)
            .landmarks
            .keys()
            .filter(|&landmark| {
                self.data
                    .landmark(reconstruction, landmark)
                    .observations
                    .len()
                    >= 2
            })
            .count();
        info!("started with {} 3d landmarks", num_3d_landmarks,);

        // Split any landmark that isnt robust.
        for landmark in landmarks {
            if !self.is_landmark_robust(reconstruction, landmark) {
                self.split_landmark(reconstruction, landmark);
            }
        }

        // Log the data after filtering.
        let num_3d_landmarks: usize = self
            .data
            .reconstruction(reconstruction)
            .landmarks
            .keys()
            .filter(|&landmark| {
                self.data
                    .landmark(reconstruction, landmark)
                    .observations
                    .len()
                    >= 2
            })
            .count();
        info!("ended with {} 3d landmarks", num_3d_landmarks,);
    }

    /// Attempts to merge two landmarks. If it succeeds, it returns the landmark ID.
    fn try_merge_landmarks(
        &mut self,
        reconstruction: ReconstructionKey,
        landmark_a: LandmarkKey,
        landmark_b: LandmarkKey,
    ) -> Option<LandmarkKey> {
        // If the same view appears in each landmark, then that means two different features from the same view
        // would appear in the resulting landmark, which is invalid.
        let duplicate_view = self
            .data
            .landmark_observations(reconstruction, landmark_a)
            .any(|(view_a, _)| {
                self.data
                    .landmark_observations(reconstruction, landmark_b)
                    .any(|(view_b, _)| view_a == view_b)
            });
        if duplicate_view {
            // We got a duplicate view, so return none.
            return None;
        }
        // Get an iterator over all the observations in both landmarks.
        let all_observations = self
            .data
            .landmark_observations(reconstruction, landmark_a)
            .chain(self.data.landmark_observations(reconstruction, landmark_b));

        // Triangulate the point which would be the combination of all landmarks.
        let point = self.triangulate_observations(reconstruction, all_observations.clone())?;

        // Determine if all observations would be good if merged.
        let all_good = all_observations.clone().all(|(view, feature)| {
            self.data.is_observation_good(
                reconstruction,
                view,
                feature,
                point,
                self.settings.merge_cosine_distance_threshold,
            )
        });
        // Non-lexical lifetimes failed me.
        drop(all_observations);

        if all_good {
            // If they would all be good, merge them.
            Some(
                self.data
                    .merge_landmarks(reconstruction, landmark_a, landmark_b),
            )
        } else {
            // If they would not all be good, dont merge them.
            None
        }
    }

    pub fn merge_nearby_landmarks(&mut self, reconstruction: ReconstructionKey) {
        use rstar::primitives::PointWithData;
        use rstar::RTree;
        type LandmarkPoint = PointWithData<LandmarkKey, [f64; 3]>;
        info!("merging reconstruction landmarks");
        // Only take landmarks with at least two observations.
        let landmarks: Vec<LandmarkPoint> = self
            .data
            .reconstruction(reconstruction)
            .landmarks
            .iter()
            .filter_map(|(landmark, _)| {
                self.triangulate_landmark(reconstruction, landmark)
                    .and_then(|wp| {
                        wp.point()
                            .map(|p| LandmarkPoint::new(landmark, [p.x, p.y, p.z]))
                    })
            })
            .collect();
        let landmark_index: RTree<LandmarkPoint> = RTree::bulk_load(landmarks.clone());

        let mut num_merged = 0usize;
        for landmark_point_a in landmarks {
            // Check if landmark a still exists.
            if self
                .data
                .reconstruction(reconstruction)
                .landmarks
                .contains_key(landmark_point_a.data)
            {
                // If the landmark still exists, search its nearest neighbors (up to 4, the first is itself).
                let position: &[f64; 3] = landmark_point_a.position();
                for landmark_point_b in landmark_index.nearest_neighbor_iter(position).take(5) {
                    // Check if it is not matched to itself, if landmark b still exists, and if merging was successful.
                    if landmark_point_a.data != landmark_point_b.data
                        && self
                            .data
                            .reconstruction(reconstruction)
                            .landmarks
                            .contains_key(landmark_point_b.data)
                        && self
                            .try_merge_landmarks(
                                reconstruction,
                                landmark_point_a.data,
                                landmark_point_b.data,
                            )
                            .is_some()
                    {
                        num_merged += 1;
                    }
                }
            }
        }
        info!("merged {} landmarks", num_merged);
    }

    /// This checks if a landmark is sufficiently robust by observing its number of robust observations and the largest
    /// observed angle of incidence to see if they are within appropriate thresholds.
    pub fn is_landmark_robust(
        &self,
        reconstruction: ReconstructionKey,
        landmark: LandmarkKey,
    ) -> bool {
        self.landmark_robust_observations(reconstruction, landmark)
            .count()
            >= self.settings.robust_minimum_observations
            && self
                .landmark_robust_observations(reconstruction, landmark)
                .map(|(view, feature)| {
                    let pose = self.data.pose(reconstruction, view).inverse();
                    pose.isometry()
                        * self
                            .data
                            .observation_keypoint(reconstruction, view, feature)
                            .bearing()
                })
                .tuple_combinations()
                .any(|(bearing_a, bearing_b)| {
                    1.0 - bearing_a.dot(&bearing_b)
                        > self.settings.incidence_minimum_cosine_distance
                })
    }

    pub fn triangulate_landmark(
        &self,
        reconstruction: ReconstructionKey,
        landmark: LandmarkKey,
    ) -> Option<WorldPoint> {
        self.triangulate_observations(
            reconstruction,
            self.data.landmark_observations(reconstruction, landmark),
        )
    }

    /// Return observations that are robust of a landmark.
    pub fn landmark_robust_observations(
        &self,
        reconstruction: ReconstructionKey,
        landmark: LandmarkKey,
    ) -> impl Iterator<Item = (ViewKey, usize)> + Clone + '_ {
        self.triangulate_landmark(reconstruction, landmark)
            .map(|point| {
                self.data.landmark_robust_observations(
                    reconstruction,
                    landmark,
                    point,
                    self.settings.robust_maximum_cosine_distance,
                )
            })
            .into_iter()
            .flatten()
    }

    /// Triangulates a landmark only if it is robust and only using robust observations.
    pub fn triangulate_landmark_robust(
        &self,
        reconstruction: ReconstructionKey,
        landmark: LandmarkKey,
    ) -> Option<WorldPoint> {
        if self.is_landmark_robust(reconstruction, landmark) {
            self.triangulate_observations(
                reconstruction,
                self.landmark_robust_observations(reconstruction, landmark),
            )
        } else {
            None
        }
    }

    pub fn triangulate_landmark_with_appended_observations(
        &self,
        reconstruction: ReconstructionKey,
        landmark: LandmarkKey,
        observations: impl Iterator<Item = (WorldToCamera, NormalizedKeyPoint)>,
    ) -> Option<WorldPoint> {
        self.triangulator.triangulate_observations(
            self.data
                .landmark_observations(reconstruction, landmark)
                .map(|(view, feature)| {
                    (
                        self.data.pose(reconstruction, view),
                        self.data
                            .observation_keypoint(reconstruction, view, feature),
                    )
                })
                .chain(observations),
        )
    }

    pub fn triangulate_landmark_with_appended_observations_and_verify(
        &self,
        reconstruction: ReconstructionKey,
        landmark: LandmarkKey,
        mut observations: impl Iterator<Item = (WorldToCamera, NormalizedKeyPoint)> + Clone,
    ) -> Option<WorldPoint> {
        self.triangulate_landmark_with_appended_observations(
            reconstruction,
            landmark,
            observations.clone(),
        )
        .filter(|world_point| {
            let verify = |pose: WorldToCamera, keypoint: NormalizedKeyPoint| {
                let camera_point = pose.transform(*world_point);
                let residual = 1.0 - keypoint.bearing().dot(&camera_point.bearing());
                residual.is_finite() && residual < self.settings.cosine_distance_threshold
            };
            self.data
                .landmark_observations(reconstruction, landmark)
                .all(|(view, feature)| {
                    let pose = self.data.pose(reconstruction, view);
                    let keypoint = self
                        .data
                        .observation_keypoint(reconstruction, view, feature);
                    verify(pose, keypoint)
                })
                && observations.all(|(pose, keypoint)| verify(pose, keypoint))
        })
    }

    /// Triangulates a landmark with observations added. An observation is a (view, feature) pair.
    pub fn triangulate_observations(
        &self,
        reconstruction: ReconstructionKey,
        observations: impl Iterator<Item = (ViewKey, usize)>,
    ) -> Option<WorldPoint> {
        self.triangulator
            .triangulate_observations(observations.map(|(view, feature)| {
                (
                    self.data.pose(reconstruction, view),
                    self.data
                        .observation_keypoint(reconstruction, view, feature),
                )
            }))
    }

    /// Use this gratuitously to help debug.
    ///
    /// This is useful when the system gets into an inconsistent state due to an internal
    /// bug. This kind of issue can't be tracked down by debugging, since you have to rewind
    /// backwards and look for connections between data to understand where the issue went wrong.
    /// By using this, you can observe errors as they accumulate in the system to better track them down.
    pub fn sanity_check(&self, reconstruction: ReconstructionKey) {
        info!("SANITY CHECK: checking to see if all view landmarks still exist");
        for view in self
            .data
            .reconstruction(reconstruction)
            .views
            .iter()
            .map(|(view, _)| view)
        {
            for (feature, &landmark) in self.data.reconstruction(reconstruction).views[view]
                .landmarks
                .iter()
                .enumerate()
            {
                if !self
                    .data
                    .reconstruction(reconstruction)
                    .landmarks
                    .contains_key(landmark)
                {
                    error!("SANITY CHECK FAILURE: landmark associated with reconstruction {:?}, view {:?}, and feature {} does not exist, it was landmark {:?}", reconstruction, view, feature, landmark);
                } else {
                    let observation = self
                        .data
                        .landmark(reconstruction, landmark)
                        .observations
                        .get(&view);
                    if observation != Some(&feature) {
                        error!("SANITY CHECK FAILURE: landmark associated with reconstruction {:?}, view {:?}, and feature {} does not contain the feature as an observation, instead found feature {:?}", reconstruction, view, feature, observation);
                    }
                }
            }
        }
        info!("SANITY CHECK ENDED");
    }
}

fn matching<'a>(
    a_descriptors: impl Iterator<Item = &'a BitArray<64>>,
    b_descriptors: impl Iterator<Item = &'a BitArray<64>> + Clone,
    better_by: u32,
) -> Vec<Option<(usize, u32)>> {
    a_descriptors
        .map(|a| {
            let mut b_descriptors = b_descriptors.clone().enumerate();
            let mut sufficiently_best = true;
            let mut best = (0usize, a.distance(b_descriptors.next()?.1));
            for (ix, b) in b_descriptors {
                let distance = a.distance(b);
                match distance.cmp(&best.1) {
                    Ordering::Less => {
                        // The match must be better than `better_by`, or it is not sufficiently the best.
                        sufficiently_best = best.1 - distance > better_by;
                        best = (ix, distance);
                    }
                    Ordering::Equal => sufficiently_best = false,
                    Ordering::Greater => {}
                }
            }
            if sufficiently_best {
                Some(best)
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
}

fn symmetric_matching_bags<'a>(
    a_frame: &'a Frame,
    a_bag: usize,
    b_frame: &'a Frame,
    b_bag: usize,
    better_by: u32,
) -> impl Iterator<Item = (FeatureMatch<usize>, u32)> + 'a {
    // Get the bag iterators for each.
    let a_bag_descriptors = a_frame.bag_descriptors(a_bag);
    let b_bag_descriptors = b_frame.bag_descriptors(b_bag);

    // Get the bag starts (the index the bag features start at in each frame).
    let a_bag_start = a_frame.bag_start(a_bag);
    let b_bag_start = b_frame.bag_start(b_bag);

    // The best match for each feature in frame a to frame b's features.
    let forward_matches = matching(
        a_bag_descriptors.clone(),
        b_bag_descriptors.clone(),
        better_by,
    );
    // The best match for each feature in frame b to frame a's features.
    let reverse_matches = matching(
        b_bag_descriptors.clone(),
        a_bag_descriptors.clone(),
        better_by,
    );
    forward_matches
        .into_iter()
        .enumerate()
        .filter_map(move |(aix, bix)| {
            // First we only proceed if there was a sufficient bix match.
            // Filter out matches which are not symmetric.
            // Symmetric is defined as the best and sufficient match of a being b,
            // and likewise the best and sufficient match of b being a.
            bix.map(|(bix, distance)| (FeatureMatch(aix, bix), distance))
                .filter(|&(FeatureMatch(aix, bix), _)| {
                    reverse_matches[bix].map(|(bix, _)| bix) == Some(aix)
                })
        })
        .map(move |(FeatureMatch(aix, bix), distance)| {
            // We need to map the feature indices into feature vector indices rather than bag indices.
            (FeatureMatch(aix + a_bag_start, bix + b_bag_start), distance)
        })
}

fn symmetric_matching<'a>(
    a_frame: &'a Frame,
    b_frame: &'a Frame,
    better_by: u32,
) -> impl Iterator<Item = FeatureMatch<usize>> + 'a {
    // Get the bag iterators for each.
    let a_descriptors = a_frame.descriptors();
    let b_descriptors = b_frame.descriptors();

    // The best match for each feature in frame a to frame b's features.
    let forward_matches = matching(a_descriptors.clone(), b_descriptors.clone(), better_by);
    // The best match for each feature in frame b to frame a's features.
    let reverse_matches = matching(b_descriptors.clone(), a_descriptors.clone(), better_by);
    forward_matches
        .into_iter()
        .enumerate()
        .filter_map(move |(aix, bix)| {
            // First we only proceed if there was a sufficient bix match.
            // Filter out matches which are not symmetric.
            // Symmetric is defined as the best and sufficient match of a being b,
            // and likewise the best and sufficient match of b being a.
            bix.map(|(bix, _)| FeatureMatch(aix, bix))
                .filter(|&FeatureMatch(aix, bix)| {
                    reverse_matches[bix].map(|(bix, _)| bix) == Some(aix)
                })
        })
}
