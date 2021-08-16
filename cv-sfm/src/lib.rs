mod bicubic;
mod codewords;
mod export;
mod settings;

pub use export::*;
pub use settings::*;

use argmin::core::{ArgminKV, ArgminOp, Error, Executor, IterState, Observe, ObserverMode};
use average::Mean;
use bitarray::{BitArray, Hamming};
use cv_core::{
    nalgebra::{IsometryMatrix3, Matrix6x2, Point3, UnitVector3, Vector3, Vector5, Vector6},
    sample_consensus::{Consensus, Estimator},
    CameraModel, CameraPoint, CameraToCamera, FeatureMatch, FeatureWorldMatch, Pose, Projective,
    TriangulatorObservations, TriangulatorRelative, WorldPoint, WorldToCamera, WorldToWorld,
};
use cv_optimize::{
    single_view_nelder_mead, spherical_to_cartesian, three_view_nelder_mead, two_view_nelder_mead,
    SingleViewOptimizer, StructurelessThreeViewOptimizer, StructurelessTwoViewOptimizer,
};
use cv_pinhole::{CameraIntrinsicsK1Distortion, EssentialMatrix};
use float_ord::FloatOrd;
use hamming_lsh::HammingHasher;
use hgg::{Hgg, HggLite};
use image::DynamicImage;
use itertools::{izip, Itertools};
use log::*;
use maplit::hashmap;
use rand::{seq::SliceRandom, Rng};
use slotmap::{new_key_type, DenseSlotMap};
use space::{Knn, KnnInsert, KnnMap, Metric};
use std::{
    cell::RefCell,
    cmp::{self, Reverse},
    collections::{HashMap, HashSet},
    mem,
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
    pub struct ConstraintKey;
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

fn canonical_view_order(mut views: [ViewKey; 3]) -> [ViewKey; 3] {
    views.sort_unstable();
    views
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Feature {
    pub keypoint: UnitVector3<f64>,
    pub color: [u8; 3],
}

#[derive(Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Frame {
    /// A VSlam::feeds index
    pub feed: FeedKey,
    /// This frame's index in the feed.
    pub feed_frame: usize,
    /// A KnnMap from feature descriptors to keypoint and color data.
    pub descriptor_features: HggLite<Hamming, BitArray<64>, Feature>,
    /// The views this frame produced.
    pub view: Option<(ReconstructionKey, ViewKey)>,
    /// The LSH of this frame.
    pub lsh: BitArray<128>,
}

impl Frame {
    pub fn feature(&self, ix: usize) -> &Feature {
        self.descriptor_features.get_value(ix).unwrap()
    }

    pub fn keypoint(&self, ix: usize) -> UnitVector3<f64> {
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
    /// Views contained in this reconstruction
    pub views: DenseSlotMap<ViewKey, View>,
    /// Landmarks contained in this reconstruction
    pub landmarks: DenseSlotMap<LandmarkKey, Landmark>,
    /// Constraints imposed during pose optimization
    pub constraints: DenseSlotMap<ConstraintKey, ThreeViewConstraint>,
    /// Views that need to have landmarks merged after optimization
    pub merge_views: Vec<ViewKey>,
}

/// Contains the results of a bundle adjust
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct BundleAdjustment {
    /// The reconstruction the bundle adjust is happening on.
    reconstruction: ReconstructionKey,
    /// Maps VSlam::views IDs to poses
    updated_views: Vec<(ViewKey, WorldToCamera)>,
    /// Views that need to be removed.
    removed_views: Vec<ViewKey>,
}

/// Contains a three-view constraint, which includes three views and transforms between them.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct ThreeViewConstraint {
    /// The three views.
    views: [ViewKey; 3],
    /// The two poses from the first pose to the second and third poses respectively.
    poses: [IsometryMatrix3<f64>; 2],
}

impl ThreeViewConstraint {
    /// The key (left) is the view which we are transforming to while the view that comes with
    /// the `CameraToCamera` is the view which we transform from.
    fn edge_constraints(&self) -> impl Iterator<Item = (ViewKey, (ViewKey, IsometryMatrix3<f64>))> {
        let views = self.views;
        let [first, second] = self.poses;
        let first_to_second = second * first.inverse();
        std::array::IntoIter::new([
            (views[0], (views[2], second.inverse())),
            (views[0], (views[1], first.inverse())),
            (views[1], (views[0], first)),
            (views[1], (views[2], first_to_second.inverse())),
            (views[2], (views[1], first_to_second)),
            (views[2], (views[0], second)),
        ])
    }
}

/// Contains several two-view relative-pose constraints for graph optimization
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Constraints {
    /// A map with the constraints for each view.
    /// The key is the view which we are transforming to while the view that comes with
    /// the `CameraToCamera` is the view which we transform from.
    edges: HashMap<ViewKey, Vec<(ViewKey, IsometryMatrix3<f64>)>>,
    /// The views which were part of a failed optimization and the number of times they failed.
    strikes: HashMap<ViewKey, usize>,
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
    lsh_to_frame: HggLite<Hamming, BitArray<128>, FrameKey>,
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

    pub fn frame_mut(&mut self, frame: FrameKey) -> &mut Frame {
        &mut self.frames[frame]
    }

    pub fn keypoint(&self, frame: FrameKey, feature: usize) -> UnitVector3<f64> {
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

    pub fn view_frame(&self, reconstruction: ReconstructionKey, view: ViewKey) -> &Frame {
        self.frame(self.view(reconstruction, view).frame)
    }

    fn view_frame_mut(&mut self, reconstruction: ReconstructionKey, view: ViewKey) -> &mut Frame {
        self.frame_mut(self.view(reconstruction, view).frame)
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
        self.color(self.view(reconstruction, view).frame, feature)
    }

    pub fn observation_keypoint(
        &self,
        reconstruction: ReconstructionKey,
        view: ViewKey,
        feature: usize,
    ) -> UnitVector3<f64> {
        self.keypoint(self.view(reconstruction, view).frame, feature)
    }

    pub fn is_observation_good(
        &self,
        reconstruction: ReconstructionKey,
        view: ViewKey,
        feature: usize,
        point: WorldPoint,
        threshold: f64,
    ) -> bool {
        is_observation_good_raw(
            self.reconstructions[reconstruction].views[view].pose,
            self.observation_keypoint(reconstruction, view, feature),
            point,
            threshold,
        )
    }

    pub fn observation_residual(
        &self,
        reconstruction: ReconstructionKey,
        view: ViewKey,
        feature: usize,
        point: WorldPoint,
    ) -> f64 {
        observation_residual_raw(
            self.reconstructions[reconstruction].views[view].pose,
            self.observation_keypoint(reconstruction, view, feature),
            point,
        )
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
        first_matches: Vec<[usize; 2]>,
        second_matches: Vec<[usize; 2]>,
    ) -> ReconstructionKey {
        // Create a new empty reconstruction.
        let reconstruction = self.reconstructions.insert(Reconstruction::default());
        // Add frame A to new reconstruction using an empty set of landmarks so all features are added as new landmarks.
        let center_view = self.add_view(reconstruction, center, Pose::identity(), |_| None);
        // Create a map for first landmarks.
        let first_landmarks: HashMap<usize, LandmarkKey> = first_matches
            .iter()
            .map(|&[c, f]| (f, c))
            .chain(combined_matches.iter().map(|&(c, f, _)| (f, c)))
            .map(|(f, c)| (f, self.observation_landmark(reconstruction, center_view, c)))
            .collect();
        // Add frame B to new reconstruction using the extracted landmark, bix pairs.
        let first_view = self.add_view(
            reconstruction,
            first,
            first_pose.isometry().into(),
            |feature| first_landmarks.get(&feature).copied(),
        );
        // Create a map for second landmarks.
        let second_landmarks: HashMap<usize, LandmarkKey> = second_matches
            .iter()
            .map(|&[c, s]| (s, c))
            .chain(combined_matches.iter().map(|&(c, _, s)| (s, c)))
            .map(|(s, c)| (s, self.observation_landmark(reconstruction, center_view, c)))
            .collect();
        // Add frame B to new reconstruction using the extracted landmark, bix pairs.
        let second_view = self.add_view(
            reconstruction,
            second,
            second_pose.isometry().into(),
            |feature| second_landmarks.get(&feature).copied(),
        );
        self.reconstructions[reconstruction]
            .constraints
            .insert(ThreeViewConstraint {
                views: canonical_view_order([center_view, first_view, second_view]),
                poses: [first_pose.isometry(), second_pose.isometry()],
            });
        self.reconstructions[reconstruction]
            .merge_views
            .extend_from_slice(&[center_view, first_view, second_view]);
        reconstruction
    }

    /// Adds a new View.
    ///
    /// `existing_landmark` is passed a Frame::features index and returns the associated landmark if it exists.
    fn add_view(
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
            updated_views,
            removed_views,
        } = bundle_adjust;
        for (view, pose) in updated_views {
            self.reconstructions[reconstruction].views[view].pose = pose;
        }
        for view in removed_views {
            info!("removing view from reconstruction");
            self.remove_view(reconstruction, view);
        }
    }

    fn remove_view(&mut self, reconstruction: ReconstructionKey, view: ViewKey) {
        self.view_frame_mut(reconstruction, view).view = None;
        let landmarks = mem::take(&mut self.reconstructions[reconstruction].views[view].landmarks);
        for landmark in landmarks {
            match self.reconstructions[reconstruction].landmarks[landmark]
                .observations
                .len()
            {
                0 => panic!("landmark had 0 observations"),
                1 => {
                    self.reconstructions[reconstruction]
                        .landmarks
                        .remove(landmark);
                }
                _ => {
                    self.reconstructions[reconstruction].landmarks[landmark]
                        .observations
                        .remove(&view);
                }
            }
        }
        self.reconstructions[reconstruction].views.remove(view);
    }

    /// Splits the observation into its own landmark.
    ///
    /// Returns `true` if the observation was split into a new landmark.
    /// Returns `false` if the landmark already only had one observation.
    fn split_observation(
        &mut self,
        reconstruction: ReconstructionKey,
        view: ViewKey,
        feature: usize,
    ) -> bool {
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
            true
        } else {
            false
        }
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
        let mut descriptor_features = HggLite::new(Hamming).insert_knn(32);
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
    C1: Consensus<PE, FeatureWorldMatch>,
    C2: Consensus<EE, FeatureMatch>,
    PE: Estimator<FeatureWorldMatch, Model = WorldToCamera>,
    EE: Estimator<FeatureMatch, Model = EssentialMatrix>,
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
                    .or_else(opeek(|| info!("failed to incorporate frame")))
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
        let view = self
            .incorporate_frame(reconstruction, frame, view_matches)
            .or_else(opeek(|| info!("failed to incorporate frame")))?;
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
        matches: impl Iterator<Item = [usize; 2]>,
        residual_threshold: f64,
        incidence_threshold: f64,
    ) -> Vec<[usize; 2]> {
        matches
            .filter(|m| {
                let FeatureMatch(a, b) = FeatureMatch(a.keypoint(m[0]), b.keypoint(m[1]));
                self.is_bi_landmark_robust(pose, a, b, residual_threshold, incidence_threshold)
            })
            .collect()
    }

    /// Checks if two observations from two views with a [`CameraToCamera`] relative pose form a robust landmark.
    ///
    /// If succesful, returns the point from the perspective of `A`.
    fn pair_robust_point(
        &self,
        pose: CameraToCamera,
        a: UnitVector3<f64>,
        b: UnitVector3<f64>,
        maximum_cosine_distance: f64,
        incidence_minimum_cosine_distance: f64,
    ) -> Option<CameraPoint> {
        let p = self.triangulator.triangulate_relative(pose, a, b)?;
        let is_cosine_distance_satisfied = 1.0 - p.bearing().dot(&a) < maximum_cosine_distance
            && 1.0 - pose.transform(p).bearing().dot(&b) < maximum_cosine_distance;
        let is_incidence_angle_satisfied =
            1.0 - (pose.isometry() * a).dot(&b) > incidence_minimum_cosine_distance;
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
        Vec<[usize; 2]>,
        Vec<[usize; 2]>,
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
        for ((first, (first_pose, first_matches)), (second, (second_pose, second_matches))) in
            options.iter().tuple_combinations()
        {
            // Create a map from the center features to the second matches.
            let second_map: HashMap<usize, usize> =
                second_matches.iter().map(|&[c, s]| (c, s)).collect();
            // Use the map created above to create triples of (center, first, second) feature matches.
            let mut common = first_matches
                .iter()
                .filter_map(|&[c, f]| Some((c, f, *second_map.get(&c)?)))
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

                    let fp = self
                        .pair_robust_point(
                            *first_pose,
                            c,
                            f,
                            self.settings
                                .three_view_relative_scale_maximum_cosine_distance,
                            self.settings
                                .robust_observation_incidence_minimum_cosine_distance,
                        )?
                        .point()?
                        .coords;
                    let sp = self
                        .pair_robust_point(
                            *second_pose,
                            c,
                            s,
                            self.settings
                                .three_view_relative_scale_maximum_cosine_distance,
                            self.settings
                                .robust_observation_incidence_minimum_cosine_distance,
                        )?
                        .point()?
                        .coords;
                    let ratio = fp.norm_squared() / sp.norm_squared();
                    if !ratio.is_normal() {
                        return None;
                    }
                    Some(ratio)
                })
                .collect_vec();
            if common_filtered_relative_scales_squared.len()
                < self.settings.three_view_minimum_relative_scales
            {
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
            let opti_matches: Vec<[UnitVector3<f64>; 3]> = common
                .iter()
                .map(|&(c, f, s)| {
                    let c = self.data.frame(center).keypoint(c);
                    let f = self.data.frame(*first).keypoint(f);
                    let s = self.data.frame(*second).keypoint(s);
                    [c, f, s]
                })
                .take(self.settings.three_view_optimization_landmarks)
                .collect::<Vec<_>>();

            info!(
                "performing Nelder-Mead optimization on both poses using {} matches out of {}",
                opti_matches.len(),
                common.len()
            );

            let loss_cutoff = self.settings.three_view_loss_cutoff;
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
                continue;
            }

            let solver = three_view_nelder_mead(first_pose, second_pose)
                .sd_tolerance(self.settings.three_view_std_dev_threshold);
            let constraint = StructurelessThreeViewOptimizer::new(
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

            first_pose = Pose::from_se3(opti_result.state.best_param.column(0).into());
            second_pose = Pose::from_se3(opti_result.state.best_param.column(1).into());

            // Create a map from the center features to the first matches.
            let first_map: HashMap<usize, usize> =
                first_matches.iter().map(|&[c, f]| (c, f)).collect();

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
                        self.settings.robust_maximum_cosine_distance,
                        0.0,
                    )
                })
                .collect_vec();

            let first_matches = first_matches
                .iter()
                .copied()
                .filter(|&[c, f]| {
                    if second_map.contains_key(&c) {
                        return false;
                    }
                    let c = self.data.frame(center).keypoint(c);
                    let f = self.data.frame(*first).keypoint(f);
                    self.is_bi_landmark_robust(
                        first_pose,
                        c,
                        f,
                        self.settings.robust_maximum_cosine_distance,
                        0.0,
                    )
                })
                .collect_vec();

            let second_matches = second_matches
                .iter()
                .copied()
                .filter(|&[c, s]| {
                    if first_map.contains_key(&c) {
                        return false;
                    }
                    let c = self.data.frame(center).keypoint(c);
                    let s = self.data.frame(*second).keypoint(s);
                    self.is_bi_landmark_robust(
                        second_pose,
                        c,
                        s,
                        self.settings.robust_maximum_cosine_distance,
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
        a: UnitVector3<f64>,
        b: UnitVector3<f64>,
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
        let ra = 1.0 - ap.bearing().dot(&a);
        let rb = 1.0 - bp.bearing().dot(&b);
        // All must satisfy the consensus threshold at worst.
        let satisfies_cosine_distance = [ra, rb]
            .iter()
            .all(|&r| r.is_finite() && r < residual_threshold);
        // The a bearing in the A camera reference frame.
        let a_bearing = a;
        // The b bearing in the A camera reference frame.
        let b_bearing = pose.inverse().isometry() * b;
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
        c: UnitVector3<f64>,
        f: UnitVector3<f64>,
        s: UnitVector3<f64>,
        residual_threshold: f64,
        incidence_threshold: f64,
    ) -> bool {
        // The triangulated point in the center camera.
        let cp = if let Some(cp) = self.triangulator.triangulate_observations_to_camera(
            c,
            std::array::IntoIter::new([
                (first_pose.isometry().into(), f),
                (second_pose.isometry().into(), s),
            ]),
        ) {
            cp
        } else {
            return false;
        };
        // Transform the point to the other cameras.
        let fp = first_pose.transform(cp);
        let sp = second_pose.transform(cp);
        // Compute the residuals for each observation.
        let rc = 1.0 - cp.bearing().dot(&c);
        let rf = 1.0 - fp.bearing().dot(&f);
        let rs = 1.0 - sp.bearing().dot(&s);
        // All must satisfy the consensus threshold at worst.
        let satisfies_cosine_distance = [rc, rf, rs]
            .iter()
            .all(|&r| r.is_finite() && r < residual_threshold);
        // The center bearing in the center camera reference frame.
        let cc = c;
        // The first bearing in the center camera reference frame.
        let cf = first_pose.inverse().isometry() * f;
        // The second bearing in the center camera reference frame.
        let cs = second_pose.inverse().isometry() * s;
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
    fn init_two_view(&self, a: FrameKey, b: FrameKey) -> Option<(CameraToCamera, Vec<[usize; 2]>)> {
        let a = self.data.frame(a);
        let b = self.data.frame(b);
        // A helper to convert an index match to a keypoint match given frame a and b.
        let match_ix_kps = |[feature_a, feature_b]: [usize; 2]| {
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
        let matches: Vec<[usize; 2]> = inliers.into_iter().map(|ix| original_matches[ix]).collect();

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
        let matches = self.camera_to_camera_match_points(
            a,
            b,
            pose,
            original_matches.iter().copied(),
            self.settings.two_view_maximum_cosine_distance,
            0.0,
        );

        // Set the loss cutoff to the regular maximum cosine distance.
        let loss_cutoff = self.settings.two_view_loss_cutoff;
        info!("optimizing pose with loss cutoff {}", loss_cutoff);

        // Create solver and constraint for two-view optimizer.
        let solver =
            two_view_nelder_mead(pose).sd_tolerance(self.settings.two_view_std_dev_threshold);
        let opti_matches = matches
            .iter()
            .copied()
            .map(match_ix_kps)
            .take(self.settings.two_view_optimization_maximum_matches)
            .collect_vec();
        let constraint = StructurelessTwoViewOptimizer::new(
            opti_matches.iter().copied(),
            self.triangulator.clone(),
        )
        .loss_cutoff(loss_cutoff);

        // The initial parameter is empty becasue nelder mead is passed its own initial parameter directly.
        let opti_result = Executor::new(constraint, solver, Vector5::zeros())
            .add_observer(OptimizationObserver, ObserverMode::Always)
            .max_iters(self.settings.two_view_patience as u64)
            .run()
            .expect("two-view optimization failed");

        info!(
                        "extracted two-view pose with mean capped cosine distance of {} in {} iterations with reason: {}",
                        opti_result.state.best_cost, opti_result.state.iter + 1, opti_result.state.termination_reason,
                    );

        let translation = spherical_to_cartesian(opti_result.state.best_param.xy());

        let pose = Pose::from_se3(
            [
                translation.x,
                translation.y,
                translation.z,
                opti_result.state.best_param[2],
                opti_result.state.best_param[3],
                opti_result.state.best_param[4],
            ]
            .into(),
        );

        // Retain sufficient matches.
        let matches = self.camera_to_camera_match_points(
            a,
            b,
            pose,
            original_matches.iter().copied(),
            self.settings.two_view_maximum_cosine_distance,
            0.0,
        );

        // Compute the number of robust matches (to ensure the views are looking from sufficiently distant positions).
        let robust_matches = self.camera_to_camera_match_points(
            a,
            b,
            pose,
            original_matches.iter().copied(),
            self.settings.two_view_maximum_cosine_distance,
            self.settings
                .robust_observation_incidence_minimum_cosine_distance,
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
        original_matches.sort_unstable_by_key(|&(landmark, _)| {
            cmp::Reverse(
                self.data
                    .landmark(reconstruction_key, landmark)
                    .observations
                    .len(),
            )
        });

        info!("found {} initial feature matches", original_matches.len());

        info!("retrieving only robust landmarks corresponding to matches");

        // Extract the FeatureWorldMatch for each of the features.
        let matches_3d: Vec<FeatureWorldMatch> = original_matches
            .iter()
            .filter_map(|&(landmark, feature)| {
                Some(FeatureWorldMatch(
                    new_frame.keypoint(feature),
                    self.triangulate_landmark_robust(reconstruction_key, landmark)?,
                ))
            })
            .take(self.settings.single_view_optimization_num_matches)
            .collect();

        if matches_3d.len() < self.settings.single_view_minimum_landmarks {
            info!(
                "only found {} robust landmarks, need {}; frame registration aborted",
                matches_3d.len(),
                self.settings.single_view_minimum_landmarks,
            );
            return None;
        }

        info!(
            "estimate the pose of the camera using {} robust landmarks",
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
        let restriction_rate = (self.settings.single_view_final_loss_cutoff
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
            let constraint = SingleViewOptimizer::new(matches_3d.clone()).loss_cutoff(loss_cutoff);

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

        let num_robust_final_matches = final_matches
            .values()
            .filter(|&&landmark| {
                self.triangulate_landmark_robust(reconstruction_key, landmark)
                    .is_some()
            })
            .count();

        let inlier_ratio = final_matches.len() as f64 / original_matches_len as f64;
        info!(
            "matches remaining after all filtering stages: {}; inlier ratio {}, robust matches {}",
            final_matches.len(),
            inlier_ratio,
            num_robust_final_matches
        );

        if inlier_ratio < self.settings.single_view_inlier_minimum_threshold {
            info!(
                "inlier ratio was less than the threshold for acceptance ({}); rejecting single-view match", self.settings.single_view_inlier_minimum_threshold
            );
            return None;
        }

        if num_robust_final_matches < self.settings.single_view_minimum_robust_landmarks {
            info!(
                "number of robust matches was less than the threshold for acceptance ({}); rejecting single-view match", self.settings.single_view_minimum_robust_landmarks
            );
            return None;
        }

        Some((pose, final_matches))
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
        let src_views: Vec<ViewKey> = self.data.reconstructions[src_reconstruction]
            .views
            .keys()
            .collect();
        let mut dest_views = vec![];
        for src_view in src_views {
            let frame = self.data.view(src_reconstruction, src_view).frame;

            // Transform the pose to go from (world b -> world a) -> camera.
            // Now the transformation goes from world b -> camera, which is correct.
            let pose = (self.data.view(src_reconstruction, src_view).pose.isometry()
                * dest_to_src_transform)
                .into();

            // Create the view.
            let dest_view = self.data.reconstructions[dest_reconstruction]
                .views
                .insert(View {
                    frame,
                    pose,
                    landmarks: vec![],
                });
            dest_views.push(dest_view);
            // Update the frame's view to point to the new view.
            self.data.frames[frame].view = Some((dest_reconstruction, dest_view));

            // Add all of the view's features to the reconstruction.
            for feature in 0..self.data.frame(frame).descriptor_features.len() {
                let src_landmark =
                    self.data
                        .observation_landmark(src_reconstruction, src_view, feature);
                // Check if the source landmark is already mapped to a destination landmark.
                let dest_landmark = if let Some(&dest_landmark) = landmark_map.get(&src_landmark) {
                    // Add this observation to the observations of this landmark.
                    self.data
                        .landmark_mut(dest_reconstruction, dest_landmark)
                        .observations
                        .insert(dest_view, feature);
                    dest_landmark
                } else {
                    // Create the landmark otherwise.
                    let dest_landmark =
                        self.data
                            .add_landmark(dest_reconstruction, dest_view, feature);
                    landmark_map.insert(src_landmark, dest_landmark);
                    dest_landmark
                };
                // Add the Reconstruction::landmark index to the feature landmarks vector for this view.
                self.data
                    .view_mut(dest_reconstruction, dest_view)
                    .landmarks
                    .push(dest_landmark);
            }
        }

        for view in dest_views {
            if !self.record_view_constraints(dest_reconstruction, view) {
                self.data.remove_view(dest_reconstruction, view);
            }
        }

        self.data.reconstructions.remove(src_reconstruction);
    }

    /// Takes a view from the reconstruction and a series of constraints.
    ///
    /// Returns the average delta in se(3) after considering all constraints.
    fn constrain_view(
        &self,
        reconstruction: ReconstructionKey,
        view: ViewKey,
        constraints: &HashMap<ViewKey, Vec<(ViewKey, IsometryMatrix3<f64>)>>,
        scale: f64,
    ) -> Option<WorldToCamera> {
        // Get the constraints for this view.
        let view_constraints = constraints.get(&view).or_else(opeek(|| {
            info!("failed to get any constraints for this view")
        }))?;
        assert!(!view_constraints.is_empty());

        // Get the transformation from the world to this view.
        let world_to_view = self.data.reconstructions[reconstruction].views[view]
            .pose
            .isometry();
        // Get the transformation from this camera space to the world.
        let view_to_world = world_to_view.inverse();

        // Compute the average delta in the lie algebra se(3).
        let net_delta_se3 = view_constraints
            .iter()
            .map(|(other_view, expected_other_to_view)| {
                // Get the transformation from the world to this other camera space.
                let world_to_other_view = self.data.reconstructions[reconstruction].views
                    [*other_view]
                    .pose
                    .isometry();
                // Compute the expected relative pose from view -> expected.
                let delta = expected_other_to_view * world_to_other_view * view_to_world;
                // TODO: We probably need to reject the whole pose if there is any failure here.
                CameraToCamera(delta).se3()
            })
            .sum::<Vector6<f64>>()
            * scale;

        if net_delta_se3.iter().any(|v| !v.is_finite()) {
            None
        } else {
            // Convert the se(3) delta back into an isometry and apply it to the existing pose to create the average expected pose.
            let net_delta = CameraToCamera::from_se3(net_delta_se3).0;
            Some(WorldToCamera(net_delta * world_to_view))
        }
    }

    /// Takes a three-view covisibility in an existing reconstruction and optimizes it.
    fn optimize_three_view(
        &self,
        reconstruction: ReconstructionKey,
        views: [ViewKey; 3],
        mut landmarks: Vec<LandmarkKey>,
    ) -> Option<ThreeViewConstraint> {
        let poses = views.map(|view| {
            self.data.reconstructions[reconstruction].views[view]
                .pose
                .isometry()
        });
        // Compute the relative poses in respect to the first pose.
        let mut first_pose = CameraToCamera(poses[1] * poses[0].inverse());
        let mut second_pose = CameraToCamera(poses[2] * poses[0].inverse());

        let original_scale = first_pose.isometry().translation.vector.norm()
            + second_pose.isometry().translation.vector.norm();
        // Shuffle the landmarks to avoid bias.
        landmarks.shuffle(&mut *self.rng.borrow_mut());
        // Sort the landmarks by their number of observations so the best ones are at the front.
        landmarks.sort_unstable_by_key(|&landmark| {
            cmp::Reverse(
                self.data
                    .landmark(reconstruction, landmark)
                    .observations
                    .len(),
            )
        });

        // Get the matches to use for optimization.
        let opti_matches: Vec<[UnitVector3<f64>; 3]> = landmarks
            .iter()
            .map(|&landmark| {
                views.map(|view| {
                    self.data
                        .view_frame(reconstruction, view)
                        .keypoint(self.data.landmark(reconstruction, landmark).observations[&view])
                })
            })
            .take(self.settings.optimization_maximum_landmarks)
            .collect::<Vec<_>>();
        info!("retrieved {} robust landmarks", landmarks.len());

        info!(
            "performing Nelder-Mead optimization on three-view constraint using {} landmarks",
            opti_matches.len()
        );
        if landmarks.len() < self.settings.optimization_minimum_landmarks {
            info!("insufficient robust landmarks, needed {} robust landmarks; rejecting view optimization", self.settings.optimization_minimum_landmarks);
            return None;
        }

        // Restrict the threshold for outliers to the robust maximum cosine distance since this pose is already optimized.
        let loss_cutoff = self.settings.optimization_loss_cutoff;
        info!("optimizing poses with loss cutoff {}", loss_cutoff);

        let solver = three_view_nelder_mead(first_pose, second_pose)
            .sd_tolerance(self.settings.optimization_std_dev_threshold);
        let constraint = StructurelessThreeViewOptimizer::new(
            opti_matches.iter().copied(),
            self.triangulator.clone(),
        )
        .loss_cutoff(loss_cutoff);

        // The initial parameter is empty becasue nelder mead is passed its own initial parameter directly.
        let opti_result = Executor::new(constraint, solver, Matrix6x2::zeros())
            .add_observer(OptimizationObserver, ObserverMode::Always)
            .max_iters(self.settings.optimization_three_view_constraint_patience as u64)
            .run()
            .expect("three-view optimization failed");

        info!(
            "extracted three-view poses with mean capped cosine distance of {} in {} iterations with reason: {}",
            opti_result.state.best_cost, opti_result.state.iter + 1, opti_result.state.termination_reason,
        );

        first_pose = Pose::from_se3(opti_result.state.best_param.column(0).into());
        second_pose = Pose::from_se3(opti_result.state.best_param.column(1).into());

        let final_scale = first_pose.isometry().translation.vector.norm()
            + second_pose.isometry().translation.vector.norm();

        let relative_scale = original_scale / final_scale;

        // Scale the poses back to their original scale.
        first_pose = first_pose.scale(relative_scale);
        second_pose = second_pose.scale(relative_scale);

        // TODO: Some kind of verification of the quality of the three-view constraint should be performed here.
        // Probably count the number of triangulated points that are sufficiently robust in terms of cosine distance,
        // then reject the constraint if it does not contain enough robust points.

        Some(ThreeViewConstraint {
            views,
            poses: [first_pose.isometry(), second_pose.isometry()],
        })
    }

    /// Attempts to track the frame in the reconstruction.
    ///
    /// Returns the pose and a vector of indices in the format (Reconstruction::landmarks, Frame::features).
    fn incorporate_frame(
        &mut self,
        reconstruction: ReconstructionKey,
        frame: FrameKey,
        view_matches: Vec<ViewKey>,
    ) -> Option<ViewKey> {
        let (pose, matches) = self
            .register_frame(reconstruction, frame, view_matches)
            .or_else(opeek(|| info!("failed to register frame")))?;

        let view = self.data.add_view(reconstruction, frame, pose, |feature| {
            matches.get(&feature).copied()
        });

        if self.record_view_constraints(reconstruction, view) {
            self.data.reconstructions[reconstruction]
                .merge_views
                .push(view);
            Some(view)
        } else {
            self.data.remove_view(reconstruction, view);
            None
        }
    }

    /// Generate and add view constraints to reconstruction.
    ///
    /// Returns `true` if enough constraints were added.
    fn record_view_constraints(
        &mut self,
        reconstruction: ReconstructionKey,
        view: ViewKey,
    ) -> bool {
        let constraints = self.generate_view_constraints(reconstruction, view);
        if constraints.len() < self.settings.optimization_minimum_new_constraints {
            return false;
        }
        for constraint in constraints {
            self.data.reconstructions[reconstruction]
                .constraints
                .insert(constraint);
        }
        true
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
        let frame = self.data.view(src_reconstruction, src_view).frame;
        let src_pose = self.data.view(src_reconstruction, src_view).pose;

        // Try to register the view into the dest reconstruction.
        let (dest_pose, matches) = self
            .register_frame(dest_reconstruction, frame, dest_view_matches)
            .or_else(opeek(|| info!("failed to register frame")))?;

        // Add the view to the dest reconstruction.
        let dest_view = self
            .data
            .add_view(dest_reconstruction, frame, dest_pose, |feature| {
                matches.get(&feature).copied()
            });

        // Try to record the view constraints. If this step fails, the merger fails.
        // Therefore, the view is removed from the reconstruction and the frame is
        // set to the correct view again.
        if !self.record_view_constraints(dest_reconstruction, dest_view) {
            self.data.remove_view(dest_reconstruction, dest_view);
            self.data.frames[frame].view = Some((src_reconstruction, src_view));
            info!("failed to record view constraints for matching frame; rejecting reconstruction match");
            return None;
        }

        // Extract the pose for the destination view.
        let dest_pose = self.data.view(dest_reconstruction, dest_view).pose;

        // At this point, the view is already in the destination reconstruction,
        // but it also exists in the source reconstruction. Extract the source reconstruction
        // landmarks for the view.
        let landmark_to_landmark: HashMap<LandmarkKey, LandmarkKey> = matches
            .iter()
            .map(|(&src_feature, &dest_landmark)| {
                (
                    self.data.view(src_reconstruction, src_view).landmarks[src_feature],
                    dest_landmark,
                )
            })
            .collect();

        // Now remove the view from the source reconstruction.
        // Rather than calling remove_view, we will just remove it directly so
        // that the frame is not reset. This should not cause any issues since
        // the source reconstruction will be destroyed shortly.
        self.data.reconstructions[src_reconstruction]
            .views
            .remove(src_view);

        // Since we are past the point of no return, also add the destination view to the merge_views
        // list so that it has its landmarks merged properly.
        self.data.reconstructions[dest_reconstruction]
            .merge_views
            .push(dest_view);

        // Create the transformation from the source to the destination reconstruction.
        let world_transform = WorldToWorld::from_camera_poses(src_pose, dest_pose);

        self.incorporate_reconstruction(
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
        let rescale_factor = mean_distance.recip();
        let transform = first_view.pose.isometry().inverse();

        // Transform the world.
        for view in self.data.reconstructions[reconstruction].views.values_mut() {
            // Transform the view pose itself.
            view.pose = (view.pose.isometry() * transform).into();
            // Rescale the pose's translation.
            view.pose.0.translation.vector *= rescale_factor;
        }
        for constraint in self.data.reconstructions[reconstruction]
            .constraints
            .values_mut()
        {
            // Scale the pose translations.
            for pose in &mut constraint.poses {
                pose.translation.vector *= rescale_factor;
            }
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
            self.filter_non_robust_observations(reconstruction)?;
            // Merge landmarks.
            for view in mem::take(&mut self.data.reconstructions[reconstruction].merge_views) {
                self.merge_view_landmarks(reconstruction, view);
            }
        }
        Some(reconstruction)
    }

    /// Optimizes reconstruction camera poses.
    pub fn bundle_adjust_reconstruction(
        &mut self,
        reconstruction: ReconstructionKey,
    ) -> Option<ReconstructionKey> {
        info!("bundle adjusting reconstruction");
        let constraints = self.flatten_constraints(reconstruction);
        for _ in 0..self.settings.optimization_iterations {
            if let Some(bundle_adjust) =
                self.compute_momentum_bundle_adjust(reconstruction, &constraints)
            {
                self.data.apply_bundle_adjust(bundle_adjust);
            } else {
                self.data.remove_reconstruction(reconstruction);
                return None;
            }
        }
        Some(reconstruction)
    }

    /// Optimizes the entire reconstruction.
    ///
    /// Use `num_landmarks` to control the number of landmarks used in optimization.
    ///
    /// Returns a series of camera
    fn compute_momentum_bundle_adjust(
        &self,
        reconstruction: ReconstructionKey,
        constraints: &HashMap<ViewKey, Vec<(ViewKey, IsometryMatrix3<f64>)>>,
    ) -> Option<BundleAdjustment> {
        let mut ba = BundleAdjustment {
            reconstruction,
            updated_views: vec![],
            removed_views: vec![],
        };
        // Extract all of the views, their current poses, and their optimized poses.
        for view in self.data.reconstruction(reconstruction).views.keys() {
            // Get the optimized pose.
            // This is where we perform the Nesterov update, as we compute the delta based
            // on the `nag_pose` rather than the old pose. We also apply the
            // optimization_convergence_rate (same as learning rate in ML) as a constant
            // multiplier by which we linearly interpolate the transformation of the camera.
            let pose = if let Some(pose) = self.constrain_view(
                reconstruction,
                view,
                constraints,
                self.settings.optimization_convergence_rate,
            ) {
                pose.0
            } else {
                ba.removed_views.push(view);
                continue;
            };

            ba.updated_views.push((view, pose.into()));
        }
        Some(ba).filter(|ba| ba.updated_views.len() >= 3)
    }

    /// Generates the constraints for a view using covisibilites.
    fn generate_view_constraints(
        &self,
        reconstruction: ReconstructionKey,
        view: ViewKey,
    ) -> Vec<ThreeViewConstraint> {
        // Get all of the covisibilities of this view by robust landmarks.
        let mut covisibilities = self.view_covisibilities(reconstruction, view);
        // Take only the covisibilities which satisfy the minimum landmarks requirement.
        covisibilities.retain(|_, landmarks| {
            landmarks.len()
                >= self
                    .settings
                    .optimization_robust_covisibility_minimum_landmarks
        });
        // Get the remaining views into a vector.
        let candidate_views = covisibilities.keys().copied().collect_vec();
        // Flip this so that we can find if a view contains a landmark.
        let mut landmark_views: HashMap<LandmarkKey, HashSet<ViewKey>> = HashMap::new();
        for (coview, landmarks) in &covisibilities {
            for &landmark in landmarks {
                landmark_views.entry(landmark).or_default().insert(*coview);
            }
        }
        // Go through every combination of these candidate views and look for ones that have a sufficient quantity of
        // covisible robust landmarks.
        let mut robust_three_view_covisibilities = candidate_views
            .iter()
            .copied()
            .tuple_combinations()
            .map(|(a, b)| {
                let covisible_landmarks = covisibilities[&a]
                    .iter()
                    .copied()
                    .filter(|landmark| landmark_views[landmark].contains(&b))
                    .collect_vec();
                (canonical_view_order([view, a, b]), covisible_landmarks)
            })
            .filter(|(_, covisible_landmarks)| {
                covisible_landmarks.len()
                    >= self
                        .settings
                        .optimization_robust_covisibility_minimum_landmarks
            })
            .collect_vec();

        // Sort the covisibilities such that the best ones show up first.
        robust_three_view_covisibilities.sort_unstable_by_key(|(_, covisible_landmarks)| {
            cmp::Reverse(covisible_landmarks.len())
        });

        // Limit the number of constraints to only the best constraints.
        robust_three_view_covisibilities
            .truncate(self.settings.optimization_maximum_three_view_constraints);

        // Compute the `ThreeViewConstraint` for each robust three view covisibility.
        robust_three_view_covisibilities
            .into_iter()
            .filter_map(move |(views, landmarks)| {
                self.optimize_three_view(reconstruction, views, landmarks)
            })
            .collect()
    }

    /// Gets the flattened constraints for a reconstruction for optimization.
    fn flatten_constraints(
        &self,
        reconstruction: ReconstructionKey,
    ) -> HashMap<ViewKey, Vec<(ViewKey, IsometryMatrix3<f64>)>> {
        let mut edges: HashMap<ViewKey, Vec<(ViewKey, IsometryMatrix3<f64>)>> = HashMap::new();
        for (view, constraint) in self.data.reconstructions[reconstruction]
            .constraints
            .values()
            .flat_map(|constraint| constraint.edge_constraints())
        {
            edges.entry(view).or_default().push(constraint);
        }
        edges
    }

    /// Get all of the covisibilities of a view (using robust landmarks).
    fn view_covisibilities(
        &self,
        reconstruction: ReconstructionKey,
        view: ViewKey,
    ) -> HashMap<ViewKey, Vec<LandmarkKey>> {
        let mut covisibilities: HashMap<ViewKey, Vec<LandmarkKey>> = HashMap::new();
        for &landmark in &self.data.reconstructions[reconstruction].views[view].landmarks {
            if self
                .triangulate_landmark_robust(reconstruction, landmark)
                .is_some()
            {
                for &coview in self.data.reconstructions[reconstruction].landmarks[landmark]
                    .observations
                    .keys()
                    .filter(|&&coview| coview != view)
                {
                    covisibilities.entry(coview).or_default().push(landmark);
                }
            }
        }
        covisibilities
    }

    /// Remove all constraints in the reconstruction and re-generate all constraints.
    pub fn regenerate_reconstruction(
        &mut self,
        reconstruction: ReconstructionKey,
    ) -> Option<ReconstructionKey> {
        // Remove all constraints.
        self.data.reconstructions[reconstruction]
            .constraints
            .clear();
        // Go through every view.
        let views = self.data.reconstructions[reconstruction]
            .views
            .keys()
            .collect_vec();
        for view in views {
            // Record the constraints for this view.
            // TODO: This might leave the graph disconnected
            // if a set of views only connect to each other and not the surrounding graph.
            // This needs to be fixed eventually.
            if !self.record_view_constraints(reconstruction, view) {
                self.data.remove_view(reconstruction, view);
            }
        }
        self.optimize_reconstruction(reconstruction)
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

    pub fn filter_non_robust_observations(
        &mut self,
        reconstruction: ReconstructionKey,
    ) -> Option<ReconstructionKey> {
        info!("filtering reconstruction observations");
        let landmarks: Vec<LandmarkKey> = self
            .data
            .reconstruction(reconstruction)
            .landmarks
            .iter()
            .map(|(lmix, _)| lmix)
            .collect();

        // Log the data before filtering.
        let initial_num_robust_landmarks: usize = self
            .data
            .reconstruction(reconstruction)
            .landmarks
            .keys()
            .filter(|&landmark| {
                self.triangulate_landmark_robust(reconstruction, landmark)
                    .is_some()
            })
            .count();
        info!(
            "started with {} robust landmarks",
            initial_num_robust_landmarks,
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
                        self.settings.robust_maximum_cosine_distance,
                    ) {
                        // If the observation is bad, we must remove it from the landmark and the view.
                        self.data.split_observation(reconstruction, view, feature);
                    }
                }
            } else {
                self.split_landmark(reconstruction, landmark);
            }
        }

        // Log the data before filtering.
        let final_num_robust_landmarks: usize = self
            .data
            .reconstruction(reconstruction)
            .landmarks
            .keys()
            .filter(|&landmark| {
                self.triangulate_landmark_robust(reconstruction, landmark)
                    .is_some()
            })
            .count();
        info!("ended with {} robust landmarks", final_num_robust_landmarks,);

        if final_num_robust_landmarks * 4 < initial_num_robust_landmarks {
            info!("since the number of robust landmarks more than quartered; rejecting reconstruction");
            self.data.remove_reconstruction(reconstruction);
            None
        } else if final_num_robust_landmarks < self.settings.minimum_robust_landmarks {
            info!("since the number of robust landmarks is less than the minimum required; rejecting reconstruction");
            self.data.remove_reconstruction(reconstruction);
            None
        } else {
            Some(reconstruction)
        }
    }

    /// Filters landmarks that arent robust.
    ///
    /// It is recommended to filter observations instead, which will filter out non-robust observations.
    /// This will remove the landmarks outright.
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
            if self
                .triangulate_landmark_robust(reconstruction, landmark)
                .is_none()
            {
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

    /// Merges two landmarks unconditionally. Returns the new landmark ID.
    fn merge_landmarks(
        &mut self,
        reconstruction: ReconstructionKey,
        landmark_a: LandmarkKey,
        landmark_b: LandmarkKey,
        point: WorldPoint,
    ) -> Option<LandmarkKey> {
        // First, we need to handle the situation where the same view is present in each landmark.
        let mut dups: Vec<(ViewKey, [usize; 2])> = vec![];
        for (&view, &observation_a) in self
            .data
            .landmark(reconstruction, landmark_a)
            .observations
            .iter()
        {
            if let Some(&observation_b) = self
                .data
                .landmark(reconstruction, landmark_b)
                .observations
                .get(&view)
            {
                dups.push((view, [observation_a, observation_b]));
            }
        }
        let mut success = true;
        for (view, observations) in dups {
            // In this case, we have two features in one view pointing to the same landmark.
            // To correct this, we will figure out which observation is closer to the point.
            // We will only keep the observation which has the lowest cosine distance.
            // Find the worst observation by the cosine distance.
            let &worst_observation = observations
                .iter()
                .max_by_key(|&&observation| {
                    FloatOrd(self.data.observation_residual(
                        reconstruction,
                        view,
                        observation,
                        point,
                    ))
                })
                .unwrap();

            // Split off the worst observation.
            if !self
                .data
                .split_observation(reconstruction, view, worst_observation)
            {
                // In this case, we would have split the very last observation out of the landmark.
                // If that is the case, then merging the landmark is pointless because
                // there would be nothing to merge.
                // We should continue splitting the observations off here for consistency, but
                // we need to mark that we will abort the merge after this or else it would panic from a duplicate.
                success = false;
            }
        }
        // Only continue if successful.
        success.then(|| ())?;
        // At this point, we are now sure that we have no duplicate views as they have been removed.
        let old_landmark = self.data.reconstructions[reconstruction]
            .landmarks
            .remove(landmark_b)
            .expect("landmark_b didnt exist");
        for (view, feature) in old_landmark.observations {
            // We must start by updating the landmark in the view for this feature.
            self.data.view_mut(reconstruction, view).landmarks[feature] = landmark_a;
            // Add the observation to landmark A.
            assert!(self
                .data
                .landmark_mut(reconstruction, landmark_a)
                .observations
                .insert(view, feature)
                .is_none());
        }
        Some(landmark_a)
    }

    pub fn merge_view_landmarks(&mut self, reconstruction: ReconstructionKey, view: ViewKey) {
        // Define the cosine distance as a metric.
        struct CosineDistance;
        impl Metric<UnitVector3<f64>> for CosineDistance {
            type Unit = u64;

            fn distance(&self, a: &UnitVector3<f64>, b: &UnitVector3<f64>) -> Self::Unit {
                // The reason the max is taken is because if the two vectors are EXACTLY equal to each other,
                // then it actually seems to produce a negative value, which is undesirable.
                let dis = 1.0 - a.dot(b);
                let dis = if dis.is_sign_negative() { 0.0 } else { dis };
                assert!(
                    dis.is_finite() && dis.is_sign_positive(),
                    "a: {}, b: {}",
                    a.into_inner(),
                    b.into_inner()
                );
                dis.to_bits()
            }
        }

        info!("merging landmarks into view");
        // Keep track of the num merged landmarks.
        let mut num_merged = 0usize;
        let mut num_removed = 0usize;
        // Get the view's pose.
        let pose = self.data.view(reconstruction, view).pose;
        // Transform all the points into the specified view's camera space and then add their bearings to an HGG.
        let mut bearings_to_landmarks: Hgg<CosineDistance, UnitVector3<f64>, LandmarkKey> =
            Hgg::new(CosineDistance);
        // Get the set of landmarks in this view.
        let current_view_landmarks: HashSet<LandmarkKey> = self
            .data
            .view(reconstruction, view)
            .landmarks
            .iter()
            .copied()
            .collect();
        // Only take landmarks that arent in the current view.
        for landmark in self.data.reconstructions[reconstruction]
            .landmarks
            .keys()
            .filter(|landmark| !current_view_landmarks.contains(landmark))
        {
            // Try to triangulate the landmark.
            if let Some(world_point) = self.triangulate_landmark(reconstruction, landmark) {
                // Transform the point into the camera's reference frame.
                let camera_point = pose.transform(world_point);
                // Extract the bearing of the camera point and add it to the HGG.
                bearings_to_landmarks.insert(camera_point.bearing(), landmark);
            }
        }
        // Go through every landmark in this view that has at least 2 observations.
        for (feature, landmark) in self
            .data
            .view(reconstruction, view)
            .landmarks
            .iter()
            .copied()
            .enumerate()
            .filter(|&(_, landmark)| {
                self.data
                    .landmark(reconstruction, landmark)
                    .observations
                    .len()
                    >= 2
            })
            .collect_vec()
        {
            // Find the nearest points in cosine distance.
            let knn = bearings_to_landmarks.knn(
                &self
                    .data
                    .observation_keypoint(reconstruction, view, feature),
                self.settings.merge_nearest_neighbors,
            );
            for nn in knn {
                // Get the landmark.
                let old_landmark = *bearings_to_landmarks.get_value(nn.index).unwrap();
                // Make sure the landmark still exists (and wasnt merged into a previous landmark).
                // Also make sure the landmark still has at least 2 observations (they can get removed).
                if !self.data.reconstructions[reconstruction]
                    .landmarks
                    .contains_key(old_landmark)
                    || self
                        .data
                        .landmark(reconstruction, old_landmark)
                        .observations
                        .len()
                        == 1
                {
                    continue;
                }
                // In this case, we want to check all of the observations of this landmark to see if they are also less than the threshold.
                if let Some(point) = self
                    .triangulate_landmark_with_appended_observations_and_verify_with_threshold(
                        reconstruction,
                        old_landmark,
                        self.settings.merge_maximum_cosine_distance,
                        self.data
                            .landmark_observations(reconstruction, landmark)
                            .map(|(view, feature)| {
                                (
                                    self.data.view(reconstruction, view).pose,
                                    self.data
                                        .observation_keypoint(reconstruction, view, feature),
                                )
                            }),
                    )
                {
                    // Merge the landmarks.
                    if self
                        .merge_landmarks(reconstruction, landmark, old_landmark, point)
                        .is_none()
                    {
                        num_removed += 1;
                        // If it fails to merge the landmarks, check if the landmark only has 1 observation.
                        if self
                            .data
                            .landmark(reconstruction, landmark)
                            .observations
                            .len()
                            == 1
                        {
                            // Since it only has one observation, the landmark is eliminated,
                            // and it can simply be ignored now.
                            break;
                        }
                    } else {
                        // Keep track of the successful merge.
                        num_merged += 1;
                    }
                }
            }
        }
        info!(
            "merged {} landmarks and removed {} landmarks",
            num_merged, num_removed
        );
    }

    pub fn triangulate_landmark(
        &self,
        reconstruction: ReconstructionKey,
        landmark: LandmarkKey,
    ) -> Option<WorldPoint> {
        if self
            .data
            .landmark(reconstruction, landmark)
            .observations
            .len()
            >= 2
        {
            self.triangulate_observations(
                reconstruction,
                self.data.landmark_observations(reconstruction, landmark),
            )
        } else {
            None
        }
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
        // Retrieve the robust observations.
        let robust_observations = self
            .landmark_robust_observations(reconstruction, landmark)
            .collect_vec();
        // Ensure we have the minimum robust observations.
        (robust_observations.len() >= self.settings.robust_minimum_observations).then(|| ())?;
        // Ensure at least two robust observations have an incidence angle between them exceeding the minimum.
        robust_observations
            .iter()
            .map(|&(view, feature)| {
                let pose = self.data.pose(reconstruction, view).inverse();
                pose.isometry()
                    * self
                        .data
                        .observation_keypoint(reconstruction, view, feature)
            })
            .tuple_combinations()
            .any(|(bearing_a, bearing_b)| {
                1.0 - bearing_a.dot(&bearing_b)
                    > self
                        .settings
                        .robust_observation_incidence_minimum_cosine_distance
            })
            .then(|| ())?;
        // Lastly, triangulate the point.
        self.triangulate_observations(reconstruction, robust_observations.iter().copied())
    }

    pub fn triangulate_landmark_with_appended_observations(
        &self,
        reconstruction: ReconstructionKey,
        landmark: LandmarkKey,
        observations: impl Iterator<Item = (WorldToCamera, UnitVector3<f64>)>,
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
        observations: impl Iterator<Item = (WorldToCamera, UnitVector3<f64>)> + Clone,
    ) -> Option<WorldPoint> {
        self.triangulate_landmark_with_appended_observations_and_verify_with_threshold(
            reconstruction,
            landmark,
            self.settings.robust_maximum_cosine_distance,
            observations,
        )
    }

    pub fn triangulate_landmark_with_appended_observations_and_verify_with_threshold(
        &self,
        reconstruction: ReconstructionKey,
        landmark: LandmarkKey,
        threshold: f64,
        mut observations: impl Iterator<Item = (WorldToCamera, UnitVector3<f64>)> + Clone,
    ) -> Option<WorldPoint> {
        self.triangulate_landmark_with_appended_observations(
            reconstruction,
            landmark,
            observations.clone(),
        )
        .filter(|world_point| {
            let verify = |pose: WorldToCamera, keypoint: UnitVector3<f64>| {
                let camera_point = pose.transform(*world_point);
                let residual = 1.0 - keypoint.dot(&camera_point.bearing());
                residual.is_finite() && residual < threshold
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

pub fn is_observation_good_raw(
    pose: WorldToCamera,
    bearing: UnitVector3<f64>,
    point: WorldPoint,
    threshold: f64,
) -> bool {
    let residual = observation_residual_raw(pose, bearing, point);
    // If the observation is finite and has a low enough residual, it is good.
    residual.is_finite() && residual < threshold
}

pub fn observation_residual_raw(
    pose: WorldToCamera,
    bearing: UnitVector3<f64>,
    point: WorldPoint,
) -> f64 {
    let view_point = pose.transform(point);
    1.0 - bearing.dot(&view_point.bearing())
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

fn symmetric_matching(a: &Frame, b: &Frame, better_by: u32) -> Vec<[usize; 2]> {
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
            bix.map(|bix| [aix, bix])
                .filter(|&[aix, bix]| reverse_matches[bix] == Some(aix))
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
