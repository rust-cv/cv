mod bicubic;
mod codewords;
mod export;
mod settings;

pub use export::*;
pub use settings::*;

use argmin::core::{
    ArgminKV, ArgminOp, Error, Executor, IterState, Observe, ObserverMode, TerminationReason,
};
use average::Mean;
use bitarray::{BitArray, Hamming};
use cv_core::{
    nalgebra::{Matrix6x2, Point3, Unit, Vector3, Vector6},
    sample_consensus::{Consensus, Estimator},
    Bearing, CameraModel, CameraPoint, CameraToCamera, FeatureMatch, FeatureWorldMatch, Pose,
    Projective, TriangulatorObservations, TriangulatorRelative, WorldPoint, WorldToCamera,
    WorldToWorld,
};
use cv_optimize::{
    many_view_nelder_mead, single_view_nelder_mead, three_view_nelder_mead, ManyViewConstraint,
    SingleViewConstraint, ThreeViewConstraint,
};
use cv_pinhole::{CameraIntrinsicsK1Distortion, EssentialMatrix, NormalizedKeyPoint};
use float_ord::FloatOrd;
use hamming_lsh::HammingHasher;
use hgg::HggLite as Hgg;
use image::DynamicImage;
use itertools::{izip, Itertools};
use log::*;
use maplit::hashmap;
use rand::{seq::SliceRandom, Rng};
use slotmap::{new_key_type, DenseSlotMap};
use space::{Knn, KnnInsert, KnnMap};
use std::{
    cell::RefCell,
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, HashMap},
    iter::once,
    ops::Sub,
    path::Path,
};

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

#[derive(Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Frame {
    /// A VSlam::feeds index
    pub feed: FeedKey,
    /// This frame's index in the feed.
    pub feed_frame: usize,
    /// A KnnMap from feature descriptors to keypoint and color data.
    pub descriptor_features: Hgg<Hamming, BitArray<64>, Feature>,
    /// The views this frame produced.
    pub view: Option<(ReconstructionKey, ViewKey)>,
    /// The LSH of this frame.
    pub lsh: BitArray<128>,
}

impl Frame {
    pub fn feature(&self, ix: usize) -> &Feature {
        self.descriptor_features.get_value(ix).unwrap()
    }

    pub fn keypoint(&self, ix: usize) -> NormalizedKeyPoint {
        self.feature(ix).keypoint
    }

    pub fn descriptor(&self, ix: usize) -> &BitArray<64> {
        self.descriptor_features.get_key(ix).unwrap()
    }

    pub fn color(&self, ix: usize) -> [u8; 3] {
        self.feature(ix).color
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
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct VSlamData {
    /// Contains the camera intrinsics for each feed
    feeds: DenseSlotMap<FeedKey, Feed>,
    /// Contains each one of the ongoing reconstructions
    reconstructions: DenseSlotMap<ReconstructionKey, Reconstruction>,
    /// Contains all the frames
    frames: DenseSlotMap<FrameKey, Frame>,
    /// Contains the LSH hasher.
    hasher: HammingHasher<64, 128>,
    /// The HGG to search descriptors for keypoint `(Reconstruction::view, Frame::features)` instances
    lsh_to_frame: Hgg<Hamming, BitArray<128>, FrameKey>,
}

impl Default for VSlamData {
    fn default() -> Self {
        Self {
            feeds: Default::default(),
            reconstructions: Default::default(),
            frames: Default::default(),
            hasher: HammingHasher::new_with_codewords(codewords::codewords()),
            lsh_to_frame: Default::default(),
        }
    }
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

    /// Add a [`Reconstruction`] from three initial frames and matches between their features.
    #[allow(clippy::too_many_arguments)]
    pub fn add_reconstruction(
        &mut self,
        center: FrameKey,
        first: FrameKey,
        second: FrameKey,
        first_pose: CameraToCamera,
        second_pose: CameraToCamera,
        combined_matches: Vec<(usize, usize, usize)>,
        first_matches: Vec<FeatureMatch<usize>>,
        second_matches: Vec<FeatureMatch<usize>>,
    ) -> ReconstructionKey {
        // Create a new empty reconstruction.
        let reconstruction = self.reconstructions.insert(Reconstruction::default());
        // Add frame A to new reconstruction using an empty set of landmarks so all features are added as new landmarks.
        let center_view = self.add_view(reconstruction, center, Pose::identity(), |_| None);
        // Create a map for first landmarks.
        let first_landmarks: HashMap<usize, LandmarkKey> = first_matches
            .iter()
            .map(|&FeatureMatch(c, f)| (f, c))
            .chain(combined_matches.iter().map(|&(c, f, _)| (f, c)))
            .map(|(f, c)| (f, self.observation_landmark(reconstruction, center_view, c)))
            .collect();
        // Add frame B to new reconstruction using the extracted landmark, bix pairs.
        self.add_view(
            reconstruction,
            first,
            first_pose.isometry().into(),
            |feature| first_landmarks.get(&feature).copied(),
        );
        // Create a map for second landmarks.
        let second_landmarks: HashMap<usize, LandmarkKey> = second_matches
            .iter()
            .map(|&FeatureMatch(c, s)| (s, c))
            .chain(combined_matches.iter().map(|&(c, _, s)| (s, c)))
            .map(|(s, c)| (s, self.observation_landmark(reconstruction, center_view, c)))
            .collect();
        // Add frame B to new reconstruction using the extracted landmark, bix pairs.
        self.add_view(
            reconstruction,
            second,
            second_pose.isometry().into(),
            |feature| second_landmarks.get(&feature).copied(),
        );
        reconstruction
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
        for feature in 0..self.frame(frame).descriptor_features.len() {
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
        let dest_to_src_transform = world_transform.isometry().inverse();
        let src_views: Vec<ViewKey> = self.reconstructions[src_reconstruction]
            .views
            .keys()
            .collect();
        for src_view in src_views {
            let frame = self.view_frame(src_reconstruction, src_view);

            // Transform the pose to go from (world b -> world a) -> camera.
            // Now the transformation goes from world b -> camera, which is correct.
            let pose = (self.view(src_reconstruction, src_view).pose.isometry()
                * dest_to_src_transform)
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
            for feature in 0..self.frame(frame).descriptor_features.len() {
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

    /// Get the most visually similar frames and the most recent frames.
    ///
    /// Automatically filters out the same frame so the frame wont match to itself.
    ///
    /// Returns (self_bag, found_view, found_bag) pairs for every reconstruction bag match and
    /// found_frame for each frame match.
    #[allow(clippy::type_complexity)]
    fn find_visually_similar_and_recent_frames(
        &self,
        frame: FrameKey,
        num_similar_frames: usize,
        num_recent_frames: usize,
        similar_recent_threshold: usize,
        similar_frames_search_num: usize,
    ) -> (HashMap<ReconstructionKey, Vec<ViewKey>>, Vec<FrameKey>) {
        info!(
            "trying to find {} visually similar frames and combine with {} recent (in past or future of feed) frames",
            num_similar_frames, num_recent_frames
        );
        let feed = self.frame(frame).feed;
        let frame_feed_ix = self.frame(frame).feed_frame;
        let recent_frames: Vec<FrameKey> = self
            .feed(feed)
            .frames
            .iter()
            .copied()
            .enumerate()
            .filter(|&(ix, recent_frame)| {
                recent_frame != frame && abs_difference(frame_feed_ix, ix) < num_recent_frames
            })
            .map(|(_, recent_frame)| recent_frame)
            .collect();
        let similar_frames = self
            .lsh_to_frame
            .knn_values(&self.frames[frame].lsh, similar_frames_search_num)
            .into_iter()
            .filter_map(|(_, &found_frame)| {
                let found_frame_feed = self.frame(found_frame).feed;
                let is_too_close = found_frame_feed == feed
                    && abs_difference(frame_feed_ix, self.frame(found_frame).feed_frame)
                        < similar_recent_threshold;
                if found_frame == frame || recent_frames.contains(&found_frame) || is_too_close {
                    None
                } else {
                    Some(found_frame)
                }
            })
            .take(num_similar_frames);
        // This will map reconstructions to frame matches.
        let mut reconstruction_frames: HashMap<ReconstructionKey, Vec<ViewKey>> = HashMap::new();
        // This contains frames with no reconstruction that are similar and their distance.
        let mut free_frames: Vec<FrameKey> = vec![];
        // Sort the matches into their respective reconstruction or into the free_frames otherwise.
        for found_frame in recent_frames.iter().copied().chain(similar_frames) {
            if let Some((reconstruction, found_view)) = self.frames[found_frame].view {
                reconstruction_frames
                    .entry(reconstruction)
                    .or_default()
                    .push(found_view);
            } else {
                free_frames.push(found_frame);
            }
        }

        let mut reconstruction_view_counts = reconstruction_frames
            .iter()
            .map(|(_, views)| views.len())
            .collect_vec();

        reconstruction_view_counts.sort_unstable_by_key(|&count| Reverse(count));

        info!(
            "found {} reconstructionless frame matches and reconstruction frame matches: {:?}",
            free_frames.len(),
            reconstruction_view_counts,
        );

        (reconstruction_frames, free_frames)
    }

    fn add_frame(&mut self, feed: FeedKey, features: Vec<(BitArray<64>, Feature)>) -> FrameKey {
        info!("adding frame with {} features", features.len());
        let lsh = self.hasher.hash_bag(features.iter().map(|(d, _)| d));
        let mut descriptor_features = Hgg::new(Hamming).insert_knn(32);
        for (descriptor, feature) in features {
            descriptor_features.insert(descriptor, feature);
        }
        let frame = self.frames.insert(Frame {
            feed,
            feed_frame: self.feeds[feed].frames.len(),
            descriptor_features,
            view: None,
            lsh,
        });
        self.lsh_to_frame.insert(lsh, frame);
        self.feeds[feed].frames.push(frame);
        frame
    }

    fn remove_reconstruction(&mut self, reconstruction: ReconstructionKey) {
        for view in self.reconstructions[reconstruction].views.values() {
            self.frames[view.frame].view = None;
        }
        self.reconstructions.remove(reconstruction);
    }
}

pub struct VSlam<C1, C2, PE, EE, T, R> {
    /// Mapping data
    pub data: VSlamData,
    /// Settings variables
    pub settings: VSlamSettings,
    /// The consensus algorithm for frame registration.
    pub single_view_consensus: RefCell<C1>,
    /// The consensus algorithm for two-view matches.
    pub two_view_consensus: RefCell<C2>,
    /// The PnP estimator
    pub pose_estimator: PE,
    /// The essential matrix estimator
    pub essential_estimator: EE,
    /// The triangulation algorithm
    pub triangulator: T,
    /// The random number generator
    pub rng: RefCell<R>,
}

impl<C1, C2, PE, EE, T, R> VSlam<C1, C2, PE, EE, T, R>
where
    C1: Consensus<PE, FeatureWorldMatch<NormalizedKeyPoint>>,
    C2: Consensus<EE, FeatureMatch<NormalizedKeyPoint>>,
    PE: Estimator<FeatureWorldMatch<NormalizedKeyPoint>, Model = WorldToCamera>,
    EE: Estimator<FeatureMatch<NormalizedKeyPoint>, Model = EssentialMatrix>,
    T: TriangulatorObservations + Clone,
    R: Rng,
{
    /// Creates an empty vSLAM reconstruction.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        data: VSlamData,
        settings: VSlamSettings,
        single_view_consensus: C1,
        two_view_consensus: C2,
        pose_estimator: PE,
        essential_estimator: EE,
        triangulator: T,
        rng: R,
    ) -> Self {
        Self {
            data,
            settings,
            single_view_consensus: RefCell::new(single_view_consensus),
            two_view_consensus: RefCell::new(two_view_consensus),
            pose_estimator,
            essential_estimator,
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
        let (reconstruction_frames, free_frames) =
            self.data.find_visually_similar_and_recent_frames(
                frame,
                self.settings.tracking_similar_frames,
                self.settings.tracking_recent_frames,
                self.settings.tracking_similar_frame_recent_threshold,
                self.settings.tracking_similar_frame_search_num,
            );

        // Try to localize this new frame with all of the similar frames.
        self.try_localize(frame, reconstruction_frames, free_frames);

        frame
    }

    /// Attempts to match a frame pair, creating a new reconstruction from a two view pair.
    ///
    /// Returns the VSlam::reconstructions ID if successful.
    fn try_init(
        &mut self,
        center: FrameKey,
        options: impl Iterator<Item = FrameKey>,
    ) -> Option<ReconstructionKey> {
        // Add the outcome.
        let (
            (first, first_pose),
            (second, second_pose),
            combined_matches,
            first_matches,
            second_matches,
        ) = self
            .init_reconstruction(center, options)
            .or_else(opeek(|| info!("failed to initialize reconstruction")))?;
        Some(self.data.add_reconstruction(
            center,
            first,
            second,
            first_pose,
            second_pose,
            combined_matches,
            first_matches,
            second_matches,
        ))
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
        reconstruction_frames: HashMap<ReconstructionKey, Vec<ViewKey>>,
        free_frames: Vec<FrameKey>,
    ) -> Option<(ReconstructionKey, ViewKey)> {
        // Sort the reconstruction frames by the number of views matched in them.
        let mut reconstruction_frames = reconstruction_frames.into_iter().collect_vec();
        reconstruction_frames.sort_unstable_by_key(|(_, views)| Reverse(views.len()));

        // Handle all the bag matches with existing reconstructions.
        for (dest_reconstruction, view_matches) in reconstruction_frames {
            if let Some((src_reconstruction, view)) = self.data.frames[frame].view {
                // The current frame is already in a reconstruction.
                if src_reconstruction != dest_reconstruction
                    && self.data.reconstruction(src_reconstruction).views.len() >= 3
                    && self.data.reconstruction(dest_reconstruction).views.len() >= 3
                {
                    // The frame is present in two separate reconstructions with at least 3 views.
                    // Try to register them together.
                    if self
                        .try_merge_reconstructions(
                            src_reconstruction,
                            view,
                            dest_reconstruction,
                            view_matches,
                        )
                        .is_some()
                    {
                        self.optimize_reconstruction(dest_reconstruction);
                    }
                }
                // Otherwise the frame was already in this reconstruction, so we can skip that.
            } else {
                // The current frame is not already in a reconstruction, so it must be incorporated.
                if self
                    .incorporate_frame(dest_reconstruction, frame, view_matches)
                    .is_some()
                {
                    // We need to optimize the reconstruction right away so that if it fails to bundle adjust
                    // that we remove the view (and reconstruction) immediately to try again.
                    self.optimize_reconstruction(dest_reconstruction);
                }
            }
        }

        // If we have no reconstruction yet, try to initialize one.
        if self.data.frames[frame].view.is_none() {
            self.try_init(frame, free_frames.iter().copied());
        }

        // If we already have a reconstruction, use that reconstruction to try and incorporate the found frames.
        if let Some((reconstruction, _)) = self.data.frames[frame].view {
            for found_frame in free_frames {
                // Skip the frame if we initialized with it.
                if self.data.frames[found_frame].view.is_some() {
                    continue;
                }
                // Try to incorporate the found frame into the reconstruction.
                self.try_localize_and_incorporate(reconstruction, found_frame)
                    .or_else(opeek(|| info!("failed to incorporate frame")))?;
            }
        }

        self.data.frames[frame].view
    }

    /// Attempts to match a view to a specific reconstruction.
    fn try_localize_and_incorporate(
        &mut self,
        reconstruction: ReconstructionKey,
        frame: FrameKey,
    ) -> Option<ViewKey> {
        // Try to find view matches that correspond to this reconstruction specifically.
        let view_matches = self
            .data
            .find_visually_similar_and_recent_frames(
                frame,
                self.settings.tracking_similar_frames,
                self.settings.tracking_recent_frames,
                self.settings.tracking_similar_frame_recent_threshold,
                self.settings.tracking_similar_frame_search_num,
            )
            .0
            .remove(&reconstruction)
            .or_else(opeek(|| {
                info!("failed to find any similar frames in the reconstruction")
            }))?;
        // Try to incorporate the frame into the reconstruction.
        let view = self.incorporate_frame(reconstruction, frame, view_matches)?;
        self.optimize_reconstruction(reconstruction)?;
        Some(view)
    }

    /// Triangulates the point of each match, filtering out matches which fail triangulation.
    ///
    /// If `final` is true, it will perform a final filtering using the `maximum_cosine_distance`,
    /// otherwise it will perform a more forgiving filtering using `consensus_threshold`.
    fn camera_to_camera_match_points(
        &self,
        a: &Frame,
        b: &Frame,
        pose: CameraToCamera,
        matches: impl Iterator<Item = FeatureMatch<usize>>,
        incidence_threshold: f64,
    ) -> Vec<FeatureMatch<usize>> {
        matches
            .filter(|m| {
                let FeatureMatch(a, b) = FeatureMatch(a.keypoint(m.0), b.keypoint(m.1));
                self.is_bi_landmark_robust(
                    pose,
                    a,
                    b,
                    self.settings.maximum_cosine_distance,
                    incidence_threshold,
                )
            })
            .collect()
    }

    /// Checks if two observations from two views with a [`CameraToCamera`] relative pose form a robust landmark.
    ///
    /// If succesful, returns the point from the perspective of `A`.
    fn pair_robust_point(
        &self,
        pose: CameraToCamera,
        a: NormalizedKeyPoint,
        b: NormalizedKeyPoint,
    ) -> Option<CameraPoint> {
        let p = self.triangulator.triangulate_relative(pose, a, b)?;
        let is_cosine_distance_satisfied = 1.0 - p.bearing().dot(&a.bearing()) + 1.0
            - pose.transform(p).bearing().dot(&b.bearing())
            < 2.0 * self.settings.maximum_cosine_distance;
        let is_incidence_angle_satisfied = 1.0 - (pose.isometry() * a.bearing()).dot(&b.bearing())
            > self
                .settings
                .robust_observation_incidence_minimum_cosine_distance;
        if is_cosine_distance_satisfied && is_incidence_angle_satisfied {
            Some(p)
        } else {
            None
        }
    }

    /// Tries to find a 3-view pair which can initialize a reconstruction.
    ///
    /// This works by trying to perform essential matrix estimation and pose optimization between the `center` frame
    /// and the `options`. For each initial pair that it finds, it will then attempt to incorporate
    /// a third frame also using essential matrix estimation and pose optimization among the remaining options.
    /// Once two matches have been performed, the translation must be scaled appropriately so that the two
    /// poses are in agreement on the scale (which is unknown). This is done by finding the top `three_view_optimization_landmarks`
    /// common matches between the frames that satisfy the `robust_observation_incidence_minimum_cosine_distance`
    /// setting and which satisfy the `robust_maximum_cosine_distance` setting.
    ///
    /// After that stage, we then triangulate each common match for each two-view pair in the center frame's
    /// reference point. The ratio between each pair's norm is computed. The ratios are then sorted to
    /// compute the median ratio. The median ratio (to filter outliers as best as possible) is used to scale
    /// the second pair's translation so that it is roughly consistent with the first pair.
    ///
    /// The last step we perform is to optimize and filter the resulting three-view reconstruction repeatedly.
    /// The optimized poses are used to compute all of the valid matches, which are then returned along with the frames and poses.
    #[allow(clippy::type_complexity)]
    fn init_reconstruction(
        &self,
        center: FrameKey,
        options: impl Iterator<Item = FrameKey>,
    ) -> Option<(
        (FrameKey, CameraToCamera),
        (FrameKey, CameraToCamera),
        Vec<(usize, usize, usize)>,
        Vec<FeatureMatch<usize>>,
        Vec<FeatureMatch<usize>>,
    )> {
        let options = options
            .filter_map(|option| {
                Some((
                    option,
                    self.init_two_view(center, option)
                        .or_else(opeek(|| info!("failed to initialize two-view")))?,
                ))
            })
            .collect_vec();
        'three_view: for (
            (first, (first_pose, first_matches)),
            (second, (second_pose, second_matches)),
        ) in options.iter().tuple_combinations()
        {
            // Create a map from the center features to the second matches.
            let second_map: HashMap<usize, usize> = second_matches
                .iter()
                .map(|&FeatureMatch(c, s)| (c, s))
                .collect();
            // Use the map created above to create triples of (center, first, second) feature matches.
            let mut common = first_matches
                .iter()
                .filter_map(|&FeatureMatch(c, f)| Some((c, f, *second_map.get(&c)?)))
                .collect_vec();
            common.shuffle(&mut *self.rng.borrow_mut());
            // Filter the common matches based on if they satisfy the criteria.
            // Extract the scale ratio for all triangulatable points in each pair.
            let mut common_filtered_relative_scales_squared = common
                .iter()
                .filter_map(|&(c, f, s)| {
                    let c = self.data.frame(center).keypoint(c);
                    let f = self.data.frame(*first).keypoint(f);
                    let s = self.data.frame(*second).keypoint(s);

                    let fp = self.pair_robust_point(*first_pose, c, f)?.point()?.coords;
                    let sp = self.pair_robust_point(*second_pose, c, s)?.point()?.coords;
                    let ratio = fp.norm_squared() / sp.norm_squared();
                    if !ratio.is_normal() {
                        return None;
                    }
                    Some(ratio)
                })
                .collect_vec();
            if common_filtered_relative_scales_squared.len() < 32 {
                info!(
                    "need at least 32 relative scales, but found {}; rejecting three-view match",
                    common_filtered_relative_scales_squared.len()
                );
                continue;
            }
            // Sort the filtered scales to find the median scale.
            common_filtered_relative_scales_squared.sort_unstable_by_key(|&f| FloatOrd(f));

            // Since the scale is currently squared, we need to take the square root of the middle scale.
            let median_scale = common_filtered_relative_scales_squared
                [common_filtered_relative_scales_squared.len() / 2]
                .sqrt();

            // Scale the second pose using the scale.
            let mut first_pose = *first_pose;
            let mut second_pose = second_pose.scale(median_scale);

            // Get the matches to use for optimization.
            // Initially, this just includes everything to avoid
            let mut opti_matches: Vec<(
                NormalizedKeyPoint,
                NormalizedKeyPoint,
                NormalizedKeyPoint,
            )> = common
                .iter()
                .map(|&(c, f, s)| {
                    let c = self.data.frame(center).keypoint(c);
                    let f = self.data.frame(*first).keypoint(f);
                    let s = self.data.frame(*second).keypoint(s);
                    (c, f, s)
                })
                .take(self.settings.three_view_optimization_landmarks)
                .collect::<Vec<_>>();

            info!(
                "performing Nelder-Mead optimization on both poses using {} matches out of {}",
                opti_matches.len(),
                common.len()
            );

            // We need to compute the rate that we must restrict the output at to reach the target
            // robust_maximum_cosine_distance by the end of our filter loop.
            let restriction_rate = (self.settings.robust_maximum_cosine_distance
                / self.settings.single_view_consensus_threshold)
                .powf((self.settings.three_view_filter_loop_iterations as f64).recip());

            for ix in 0..self.settings.three_view_filter_loop_iterations {
                // This gradually restricts the threshold for outliers on each iteration.
                let restriction = restriction_rate.powi(ix as i32 + 1);
                let loss_cutoff = self.settings.single_view_consensus_threshold * restriction;
                info!("optimizing poses with loss cutoff {}", loss_cutoff);

                info!(
                    "performing Nelder-Mead optimization on poses using {} robust three-way matches out of {}",
                    opti_matches.len(),
                    common.len()
                );
                if opti_matches.len() < 32 {
                    info!(
                        "need {} robust three-way matches; rejecting three-view match",
                        32
                    );
                    continue 'three_view;
                }

                let solver = three_view_nelder_mead(first_pose, second_pose)
                    .sd_tolerance(self.settings.three_view_std_dev_threshold);
                let constraint = ThreeViewConstraint::new(
                    opti_matches.iter().copied(),
                    self.triangulator.clone(),
                )
                .loss_cutoff(loss_cutoff);

                // The initial parameter is empty becasue nelder mead is passed its own initial parameter directly.
                let opti_result = Executor::new(constraint, solver, Matrix6x2::zeros())
                    .add_observer(OptimizationObserver, ObserverMode::Always)
                    .max_iters(self.settings.three_view_patience as u64)
                    .run()
                    .expect("three-view optimization failed");

                info!(
                    "extracted three-view poses with mean capped cosine distance of {} in {} iterations with reason: {}",
                    opti_result.state.best_cost, opti_result.state.iter + 1, opti_result.state.termination_reason,
                );

                if opti_result.state.termination_reason != TerminationReason::TargetToleranceReached
                {
                    info!("didn't terminate due to reaching our desired tollerance; rejecting three-view match");
                    continue 'three_view;
                }

                first_pose = Pose::from_se3(opti_result.state.best_param.column(0).into());
                second_pose = Pose::from_se3(opti_result.state.best_param.column(1).into());

                if ix != self.settings.three_view_filter_loop_iterations - 1 {
                    // Filter the common matches based on if they satisfy the criteria.

                    opti_matches = common
                        .iter()
                        .map(|&(c, f, s)| {
                            let c = self.data.frame(center).keypoint(c);
                            let f = self.data.frame(*first).keypoint(f);
                            let s = self.data.frame(*second).keypoint(s);
                            (c, f, s)
                        })
                        .filter(|&(c, f, s)| {
                            self.is_tri_landmark_robust(
                                first_pose,
                                second_pose,
                                c,
                                f,
                                s,
                                self.settings.single_view_consensus_threshold,
                                self.settings
                                    .robust_observation_incidence_minimum_cosine_distance,
                            )
                        })
                        .take(self.settings.three_view_optimization_landmarks)
                        .collect::<Vec<_>>();
                }
            }

            // Create a map from the center features to the first matches.
            let first_map: HashMap<usize, usize> = first_matches
                .iter()
                .map(|&FeatureMatch(c, f)| (c, f))
                .collect();

            let combined_matches = common
                .iter()
                .copied()
                .filter(|&(c, f, s)| {
                    let c = self.data.frame(center).keypoint(c);
                    let f = self.data.frame(*first).keypoint(f);
                    let s = self.data.frame(*second).keypoint(s);
                    self.is_tri_landmark_robust(
                        first_pose,
                        second_pose,
                        c,
                        f,
                        s,
                        self.settings.maximum_cosine_distance,
                        0.0,
                    )
                })
                .collect_vec();

            let first_matches = first_matches
                .iter()
                .copied()
                .filter(|&FeatureMatch(c, f)| {
                    if second_map.contains_key(&c) {
                        return false;
                    }
                    let c = self.data.frame(center).keypoint(c);
                    let f = self.data.frame(*first).keypoint(f);
                    self.is_bi_landmark_robust(
                        first_pose,
                        c,
                        f,
                        self.settings.maximum_cosine_distance,
                        0.0,
                    )
                })
                .collect_vec();

            let second_matches = second_matches
                .iter()
                .copied()
                .filter(|&FeatureMatch(c, s)| {
                    if first_map.contains_key(&c) {
                        return false;
                    }
                    let c = self.data.frame(center).keypoint(c);
                    let s = self.data.frame(*second).keypoint(s);
                    self.is_bi_landmark_robust(
                        second_pose,
                        c,
                        s,
                        self.settings.maximum_cosine_distance,
                        0.0,
                    )
                })
                .collect_vec();

            let num_robust_matches = common
                .iter()
                .map(|&(c, f, s)| {
                    let c = self.data.frame(center).keypoint(c);
                    let f = self.data.frame(*first).keypoint(f);
                    let s = self.data.frame(*second).keypoint(s);
                    (c, f, s)
                })
                .filter(|&(c, f, s)| {
                    self.is_tri_landmark_robust(
                        first_pose,
                        second_pose,
                        c,
                        f,
                        s,
                        self.settings.robust_maximum_cosine_distance,
                        self.settings
                            .robust_observation_incidence_minimum_cosine_distance,
                    )
                })
                .count();

            let inlier_ratio = combined_matches.len() as f64 / common.len() as f64;

            info!(
                "found {} tri-matches, {} tri-match inlier ratio, {} robust tri-matches, {} center-first matches, and {} center-second matches",
                combined_matches.len(),
                inlier_ratio,
                num_robust_matches,
                first_matches.len(),
                second_matches.len()
            );

            if num_robust_matches < self.settings.three_view_minimum_robust_matches {
                info!(
                    "need {} robust three-way matches; rejecting three-view match",
                    self.settings.three_view_minimum_robust_matches
                );
                continue;
            }

            if inlier_ratio < self.settings.three_view_inlier_ratio_threshold {
                info!(
                    "didn't reach inlier ratio of {}; rejecting three-view match",
                    self.settings.three_view_inlier_ratio_threshold
                );
                continue;
            }

            return Some((
                (*first, first_pose),
                (*second, second_pose),
                combined_matches,
                first_matches,
                second_matches,
            ));
        }
        info!("no three-view match was found among the two-view match options");
        None
    }

    fn is_bi_landmark_robust(
        &self,
        pose: CameraToCamera,
        a: NormalizedKeyPoint,
        b: NormalizedKeyPoint,
        residual_threshold: f64,
        incidence_threshold: f64,
    ) -> bool {
        // The triangulated point in the center camera.
        let ap = if let Some(ap) = self.triangulator.triangulate_relative(pose, a, b) {
            ap
        } else {
            return false;
        };
        // Transform the point to the other camera.
        let bp = pose.transform(ap);
        // Compute the residuals for each observation.
        let ra = 1.0 - ap.bearing().dot(&a.bearing());
        let rb = 1.0 - bp.bearing().dot(&b.bearing());
        // All must satisfy the consensus threshold at worst.
        let satisfies_cosine_distance = [ra, rb]
            .iter()
            .all(|&r| r.is_finite() && r < residual_threshold);
        // The a bearing in the A camera reference frame.
        let a_bearing = a.bearing();
        // The b bearing in the A camera reference frame.
        let b_bearing = pose.inverse().isometry() * b.bearing();
        // Incidence from center to first.
        let iab = 1.0 - a_bearing.dot(&b_bearing);
        // At least one pair must have a sufficiently large incidence angle.
        let satisfies_incidence_angle = iab > incidence_threshold;
        satisfies_cosine_distance && satisfies_incidence_angle
    }

    #[allow(clippy::too_many_arguments)]
    fn is_tri_landmark_robust(
        &self,
        first_pose: CameraToCamera,
        second_pose: CameraToCamera,
        c: NormalizedKeyPoint,
        f: NormalizedKeyPoint,
        s: NormalizedKeyPoint,
        residual_threshold: f64,
        incidence_threshold: f64,
    ) -> bool {
        // The triangulated point in the center camera.
        let cp = CameraPoint(
            if let Some(cp) = self.triangulator.triangulate_observations(
                once((WorldToCamera::identity(), c))
                    .chain(once((first_pose.isometry().into(), f)))
                    .chain(once((second_pose.isometry().into(), s))),
            ) {
                cp.0
            } else {
                return false;
            },
        );
        // Transform the point to the other cameras.
        let fp = first_pose.transform(cp);
        let sp = second_pose.transform(cp);
        // Compute the residuals for each observation.
        let rc = 1.0 - cp.bearing().dot(&c.bearing());
        let rf = 1.0 - fp.bearing().dot(&f.bearing());
        let rs = 1.0 - sp.bearing().dot(&s.bearing());
        // All must satisfy the consensus threshold at worst.
        let satisfies_cosine_distance = [rc, rf, rs]
            .iter()
            .all(|&r| r.is_finite() && r < residual_threshold);
        // The center bearing in the center camera reference frame.
        let cc = c.bearing();
        // The first bearing in the center camera reference frame.
        let cf = first_pose.inverse().isometry() * f.bearing();
        // The second bearing in the center camera reference frame.
        let cs = second_pose.inverse().isometry() * s.bearing();
        // Incidence from center to first.
        let icf = 1.0 - cc.dot(&cf);
        // Incidence from first to second.
        let ifs = 1.0 - cf.dot(&cs);
        // Incidence from second to center.
        let isc = 1.0 - cs.dot(&cc);
        // At least one pair must have a sufficiently large incidence angle.
        let satisfies_incidence_angle = [icf, ifs, isc]
            .iter()
            .any(|&incidence| incidence > incidence_threshold);
        satisfies_cosine_distance && satisfies_incidence_angle
    }

    /// This estimates and optimizes the [`CameraToCamera`] between frames `a` and `b` using the essential matrix estimator.
    ///
    /// It then attempts to add one of the [``]
    ///
    /// This method resolves to an undefined scale, and thus is only appropriate for initialization.
    fn init_two_view(
        &self,
        a: FrameKey,
        b: FrameKey,
    ) -> Option<(CameraToCamera, Vec<FeatureMatch<usize>>)> {
        let a = self.data.frame(a);
        let b = self.data.frame(b);
        // A helper to convert an index match to a keypoint match given frame a and b.
        let match_ix_kps = |FeatureMatch(feature_a, feature_b)| {
            FeatureMatch(a.keypoint(feature_a), b.keypoint(feature_b))
        };

        info!(
            "performing matching between {} and {} features",
            a.descriptor_features.len(),
            b.descriptor_features.len(),
        );
        // Retrieve the matches which agree with each other from each frame and filter out ones that aren't within the match threshold.
        let mut original_matches = symmetric_matching(a, b, self.settings.two_view_match_better_by);

        info!(
            "shuffle {} matches before consensus process",
            original_matches.len()
        );

        original_matches.shuffle(&mut *self.rng.borrow_mut());

        info!(
            "perform sample consensus to estimate essential matrix and filter outliers on {} matches",
            original_matches.len()
        );

        // Estimate the essential matrix and retrieve the inliers
        let (essential, inliers) = self
            .two_view_consensus
            .borrow_mut()
            .model_inliers(
                &self.essential_estimator,
                original_matches
                    .iter()
                    .copied()
                    .map(match_ix_kps)
                    .collect::<Vec<_>>()
                    .iter()
                    .copied(),
            )
            .or_else(opeek(|| {
                info!("failed to find essential matrix via consensus")
            }))?;

        // Reconstitute only the inlier matches into a matches vector.
        let matches: Vec<FeatureMatch<usize>> =
            inliers.into_iter().map(|ix| original_matches[ix]).collect();

        info!(
            "solve pose from essential matrix using {} inlier matches",
            matches.len()
        );

        // Solve the pose from the four possible poses using the given data.
        let pose = essential
            .pose_solver()
            .solve_unscaled(matches.iter().copied().map(match_ix_kps))
            .or_else(opeek(|| {
                info!("failed to solve camera pose from essential matrix")
            }))?;

        // Retain sufficient matches.
        let matches =
            self.camera_to_camera_match_points(a, b, pose, original_matches.iter().copied(), 0.0);

        // Compute the number of robust matches (to ensure the views are looking from sufficiently distant positions).
        let robust_matches = self.camera_to_camera_match_points(
            a,
            b,
            pose,
            original_matches.iter().copied(),
            self.settings.two_view_consensus_threshold,
        );

        let inlier_ratio = matches.len() as f64 / original_matches.len() as f64;
        info!(
            "estimated pose matches {} and robust matches {}; inlier ratio {}",
            matches.len(),
            robust_matches.len(),
            inlier_ratio
        );

        if robust_matches.len() < self.settings.two_view_minimum_robust_matches {
            info!(
                "only found {} robust matches, but needed {}; rejecting two-view match",
                robust_matches.len(),
                self.settings.two_view_minimum_robust_matches
            );
            return None;
        }

        if inlier_ratio < self.settings.two_view_inlier_minimum_threshold {
            info!(
                "inlier ratio was {}, but it must be above {}; rejecting two-view match",
                inlier_ratio, self.settings.two_view_inlier_minimum_threshold
            );
            return None;
        }

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
        view_matches: Vec<ViewKey>,
    ) -> Option<(WorldToCamera, HashMap<usize, LandmarkKey>)> {
        info!("trying to register frame into existing reconstruction");
        let reconstruction = self.data.reconstruction(reconstruction_key);
        let new_frame = self.data.frame(new_frame_key);

        info!("performing matching against {} views", view_matches.len());

        let mut original_matches: Vec<(LandmarkKey, usize)> = vec![];
        for self_feature in 0..new_frame.descriptor_features.len() {
            // Get the self feature descriptor.
            let self_descriptor = new_frame.descriptor(self_feature);
            // Find the top 2 features in every view match, and collect those together.
            let lm_matches = view_matches
                .iter()
                .flat_map(|&view_match| {
                    let frame_match = reconstruction.views[view_match].frame;
                    self.data.frames[frame_match]
                        .descriptor_features
                        .knn(self_descriptor, 2)
                        .into_iter()
                        .map(move |n| {
                            (
                                reconstruction.views[view_match].landmarks[n.index],
                                n.distance,
                            )
                        })
                })
                .collect_vec();

            // Find the top 2 landmark matches overall.
            // Create an array where the best items will go.
            // Note that these two matches come from the same frame and therefore are two different landmarks.
            let mut best = [lm_matches[0], lm_matches[1]];
            // Swap the items if they are in the incorrect order.
            if best[0].1 > best[1].1 {
                best.rotate_right(1);
            }

            // Iterate over each additional match.
            for &(lm, distance) in &lm_matches[2..] {
                // If its better than the worst.
                if distance < best[1].1 {
                    // Check if it is the same landmark as the best.
                    if best[0].0 == lm {
                        // In this case, it should not replace the second best, but it should replace
                        // the first best if it is better.
                        if distance < best[0].1 {
                            best[0].1 = distance;
                        }
                    } else {
                        // In this case, it isn't the same landmark at the best, so this match should
                        // replace the second best at least.
                        best[1] = (lm, distance);
                        // If it is also better than the best landmark match.
                        if distance < best[0].1 {
                            // Swap the newly added landmark and the best.
                            best.rotate_right(1);
                        }
                    }
                }
            }

            // Check if we satisfy the matching constraint.
            if best[0].1 + self.settings.single_view_match_better_by <= best[1].1 {
                original_matches.push((best[0].0, self_feature));
            }
        }

        info!(
            "original matches before filtering duplicates {}",
            original_matches.len()
        );

        let landmark_counts = original_matches.iter().counts_by(|&(landmark, _)| landmark);
        original_matches.retain(|(landmark, _)| landmark_counts[landmark] == 1);
        original_matches.shuffle(&mut *self.rng.borrow_mut());

        info!("found {} initial feature matches", original_matches.len());

        let create_3d_matches = |robust| {
            original_matches
                .iter()
                .filter_map(|&(landmark, feature)| {
                    Some(FeatureWorldMatch(
                        new_frame.keypoint(feature),
                        if robust {
                            self.triangulate_landmark_robust(reconstruction_key, landmark)?
                        } else {
                            self.triangulate_landmark(reconstruction_key, landmark)?
                        },
                    ))
                })
                .take(self.settings.single_view_optimization_num_matches)
                .collect()
        };

        info!("retrieving only robust landmarks corresponding to matches");

        // Extract the FeatureWorldMatch for each of the features.
        let matches_3d: Vec<FeatureWorldMatch<NormalizedKeyPoint>> = create_3d_matches(true);

        let matches_3d = if matches_3d.len() < self.settings.single_view_minimum_landmarks {
            info!(
                "only found {} robust triangulatable landmarks, need {}; trying non-robust landmarks",
                matches_3d.len(),
                self.settings.single_view_minimum_landmarks,
            );
            create_3d_matches(false)
        } else {
            matches_3d
        };

        if matches_3d.len() < self.settings.single_view_minimum_landmarks {
            info!(
                "only found {} triangulatable landmarks, need {}; frame registration aborted",
                matches_3d.len(),
                self.settings.single_view_minimum_landmarks,
            );
            return None;
        }

        info!(
            "estimate the pose of the camera using {} triangulatable landmarks",
            matches_3d.len()
        );

        // Estimate the pose and retrieve the inliers.
        let mut pose = self
            .single_view_consensus
            .borrow_mut()
            .model(&self.pose_estimator, matches_3d.iter().copied())
            .or_else(opeek(|| info!("failed to find view pose via consensus")))?;

        // We need to compute the rate that we must restrict the output at to reach the target
        // robust_maximum_cosine_distance by the end of our filter loop.
        let restriction_rate = (self.settings.robust_maximum_cosine_distance
            / self.settings.single_view_consensus_threshold)
            .powf((self.settings.single_view_filter_loop_iterations as f64).recip());

        for ix in 0..self.settings.single_view_filter_loop_iterations {
            // This gradually restricts the threshold for outliers on each iteration.
            let restriction = restriction_rate.powi(ix as i32 + 1);
            let loss_cutoff = self.settings.single_view_consensus_threshold * restriction;
            info!("optimizing pose with loss cutoff {}", loss_cutoff);

            // Create solver and constraint for single-view optimizer.
            let solver = single_view_nelder_mead(pose)
                .sd_tolerance(self.settings.single_view_std_dev_threshold);
            let constraint = SingleViewConstraint::new(matches_3d.clone()).loss_cutoff(loss_cutoff);

            // The initial parameter is empty becasue nelder mead is passed its own initial parameter directly.
            let opti_result = Executor::new(constraint, solver, Vector6::zeros())
                .add_observer(OptimizationObserver, ObserverMode::Always)
                .max_iters(self.settings.single_view_patience as u64)
                .run()
                .expect("single-view optimization failed");

            info!(
                "extracted single-view pose with mean capped cosine distance of {} in {} iterations with reason: {}",
                opti_result.state.best_cost, opti_result.state.iter + 1, opti_result.state.termination_reason,
            );

            if opti_result.state.termination_reason != TerminationReason::TargetToleranceReached {
                info!("didn't terminate due to reaching our desired tollerance; rejecting single-view match");
                return None;
            }

            pose = Pose::from_se3(opti_result.state.best_param);
        }

        let original_matches_len = original_matches.len();
        let final_matches: HashMap<usize, LandmarkKey> = original_matches
            .into_iter()
            .filter(|&(landmark, feature)| {
                let keypoint = new_frame.keypoint(feature);
                self.triangulate_landmark_with_appended_observations_and_verify(
                    reconstruction_key,
                    landmark,
                    std::iter::once((pose, keypoint)),
                )
                .is_some()
            })
            .map(|(landmark, feature)| (feature, landmark))
            .collect();

        let inlier_ratio = final_matches.len() as f64 / original_matches_len as f64;
        info!(
            "matches remaining after all filtering stages: {}; inlier ratio {}",
            final_matches.len(),
            inlier_ratio
        );

        if inlier_ratio < self.settings.single_view_inlier_minimum_threshold {
            info!(
                "inlier ratio was less than the threshold for acceptance ({}); rejecting single-view match", self.settings.single_view_inlier_minimum_threshold
            );
            return None;
        }

        Some((pose, final_matches))
    }

    /// Attempts to track the frame in the reconstruction.
    ///
    /// Returns the pose and a vector of indices in the format (Reconstruction::landmarks, Frame::features).
    fn incorporate_frame(
        &mut self,
        reconstruction: ReconstructionKey,
        new_frame: FrameKey,
        view_matches: Vec<ViewKey>,
    ) -> Option<ViewKey> {
        let (pose, matches) = self
            .register_frame(reconstruction, new_frame, view_matches)
            .or_else(opeek(|| info!("failed to register frame")))?;

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
        dest_view_matches: Vec<ViewKey>,
    ) -> Option<ReconstructionKey> {
        info!(
            "merging two reconstructions with source {} views and destination {} views",
            self.data.reconstructions[src_reconstruction].views.len(),
            self.data.reconstructions[dest_reconstruction].views.len()
        );
        let src_frame = self.data.view_frame(src_reconstruction, src_view);
        // Register the frame in the source reconstruction to the destination reconstruction.
        let (pose, matches) = self
            .register_frame(dest_reconstruction, src_frame, dest_view_matches)
            .or_else(opeek(|| info!("failed to register frame")))?;

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

        info!("merging completed successfully");

        Some(dest_reconstruction)
    }

    fn kps_descriptors(
        &self,
        intrinsics: &CameraIntrinsicsK1Distortion,
        image: &DynamicImage,
    ) -> Vec<(BitArray<64>, Feature)> {
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
        .map(|(keypoint, descriptor, color)| (descriptor, Feature { keypoint, color }))
        .collect()
    }

    /// This will take the first view of the reconstruction and scale everything so that the
    /// mean robust point distance from the first view is of unit length and the first view
    /// is the origin of the reconstruction.
    pub fn normalize_reconstruction(&mut self, reconstruction: ReconstructionKey) {
        // Get the first view.
        let first_view = self
            .data
            .reconstruction(reconstruction)
            .views
            .values()
            .next()
            .unwrap();
        // Compute the mean distance from the camera.
        let mean_distance = first_view
            .landmarks
            .iter()
            .filter_map(|&landmark| self.triangulate_landmark_robust(reconstruction, landmark))
            .filter_map(|wp| Some(first_view.pose.transform(wp).point()?.coords.norm()))
            .collect::<Mean>()
            .mean();

        if !mean_distance.is_normal() {
            return;
        }

        // Get the world transformation.
        let transform = first_view
            .pose
            .scale(mean_distance.recip())
            .inverse()
            .isometry();

        // Transform the world.
        for view in self.data.reconstructions[reconstruction].views.values_mut() {
            view.pose = (view.pose.isometry() * transform).into();
        }
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
        // Output the cameras.
        let cameras = self
            .data
            .reconstruction(reconstruction)
            .views
            .values()
            .map(|v| {
                let mean_distance = v
                    .landmarks
                    .iter()
                    .filter_map(|&landmark| {
                        self.triangulate_landmark_robust(reconstruction, landmark)
                    })
                    .filter_map(|wp| Some(v.pose.transform(wp).point()?.coords.norm()))
                    .collect::<Mean>()
                    .mean();
                let c2w = v.pose.inverse();
                ExportCamera {
                    optical_center: c2w.isometry() * Point3::origin(),
                    up_direction: c2w.isometry() * Vector3::y(),
                    forward_direction: c2w.isometry() * Vector3::z(),
                    focal_length: mean_distance * 0.01,
                }
            })
            .collect_vec();
        crate::export::export(
            std::fs::File::create(path).unwrap(),
            points_and_colors,
            cameras,
        );
    }

    /// Runs bundle adjustment (camera pose optimization), landmark filtering, and landmark merging.
    pub fn optimize_reconstruction(
        &mut self,
        reconstruction: ReconstructionKey,
    ) -> Option<ReconstructionKey> {
        for _ in 0..self.settings.reconstruction_optimization_iterations {
            // If there are three or more views, run global bundle-adjust.
            self.bundle_adjust_reconstruction(reconstruction)
                .or_else(opeek(|| info!("failed to bundle adjust reconstruction")))?;
            // Filter observations after running bundle-adjust.
            self.filter_observations(reconstruction);
            // Merge landmarks.
            self.merge_nearby_landmarks(reconstruction);
        }
        Some(reconstruction)
    }

    /// Optimizes reconstruction camera poses.
    pub fn bundle_adjust_reconstruction(
        &mut self,
        reconstruction: ReconstructionKey,
    ) -> Option<ReconstructionKey> {
        if let Some(bundle_adjust) = self.compute_bundle_adjust(reconstruction) {
            self.data.apply_bundle_adjust(bundle_adjust);
            Some(reconstruction)
        } else {
            self.data.remove_reconstruction(reconstruction);
            None
        }
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
    fn compute_bundle_adjust(&self, reconstruction: ReconstructionKey) -> Option<BundleAdjustment> {
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
                    return None;
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
            .loss_cutoff(self.settings.maximum_cosine_distance);

            let patience = if views.len() == 3 {
                self.settings.three_view_patience as u64
            } else {
                self.settings.many_view_patience as u64
            };

            // The initial parameter is empty becasue nelder mead is passed its own initial parameter directly.
            let opti_state = Executor::new(constraint, solver, vec![])
                .add_observer(OptimizationObserver, ObserverMode::Always)
                .max_iters(patience)
                .run()
                .expect("many-view optimization failed")
                .state;

            info!(
                "extracted poses with mean capped cosine distance of {} in {} iterations finished with reason: {}",
                opti_state.best_cost, opti_state.iter + 1, opti_state.termination_reason
            );

            if opti_state.termination_reason != TerminationReason::TargetToleranceReached {
                info!("did not reach the desired tolerance; rejecting bundle-adjust");
                return None;
            }

            let poses: Vec<WorldToCamera> = opti_state
                .best_param
                .iter()
                .map(|arr| Pose::from_se3(Vector6::from_row_slice(arr)))
                .collect();

            Some(BundleAdjustment {
                reconstruction,
                poses: views.iter().copied().zip(poses).collect(),
            })
        } else {
            warn!(
                "tried to bundle adjust reconstruction with no landmarks, which should not exist"
            );
            None
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
                        self.settings.maximum_cosine_distance,
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
                self.settings.merge_maximum_cosine_distance,
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
        use rstar::{primitives::PointWithData, RTree};
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
                        > self
                            .settings
                            .robust_observation_incidence_minimum_cosine_distance
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
                residual.is_finite() && residual < self.settings.maximum_cosine_distance
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

fn matching(a_frame: &Frame, b_frame: &Frame, better_by: u32) -> Vec<Option<usize>> {
    // If there arent at least 2 features in both frames, we produce no matches.
    if a_frame.descriptor_features.len() < 2 || b_frame.descriptor_features.len() < 2 {
        return vec![];
    }
    (0..a_frame.descriptor_features.len())
        .map(|a_feature| {
            let knn = b_frame
                .descriptor_features
                .knn(a_frame.descriptor(a_feature), 2);
            if knn[0].distance + better_by <= knn[1].distance {
                Some(knn[0].index)
            } else {
                None
            }
        })
        .collect()
}

fn symmetric_matching(a: &Frame, b: &Frame, better_by: u32) -> Vec<FeatureMatch<usize>> {
    // The best match for each feature in frame a to frame b's features.
    let forward_matches = matching(a, b, better_by);
    // The best match for each feature in frame b to frame a's features.
    let reverse_matches = matching(b, a, better_by);
    forward_matches
        .into_iter()
        .enumerate()
        .filter_map(move |(aix, bix)| {
            // First we only proceed if there was a sufficient bix match.
            // Filter out matches which are not symmetric.
            // Symmetric is defined as the best and sufficient match of a being b,
            // and likewise the best and sufficient match of b being a.
            bix.map(|bix| FeatureMatch(aix, bix))
                .filter(|&FeatureMatch(aix, bix)| reverse_matches[bix] == Some(aix))
        })
        .collect()
}

fn abs_difference<T: Sub<Output = T> + Ord>(x: T, y: T) -> T {
    if x < y {
        y - x
    } else {
        x - y
    }
}

/// Used with [`Option::or_else`] to "peek" when there is `None` in an option.
fn opeek<T>(mut f: impl FnMut()) -> impl FnOnce() -> Option<T> {
    move || {
        f();
        None
    }
}
