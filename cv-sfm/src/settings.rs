#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

/// The settings for the VSlam process.
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Debug, Copy, Clone)]
pub struct VSlamSettings {
    /// The threshold used for akaze
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_akaze_threshold")
    )]
    pub akaze_threshold: f64,
    /// The maximum cosine distance of an observation for it to be considered robust enough for optimization
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_maximum_cosine_distance")
    )]
    pub maximum_cosine_distance: f64,
    /// The maximum sine distance of a two-view observation for it to exist
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_maximum_sine_distance")
    )]
    pub maximum_sine_distance: f64,
    /// The the minimum number of robust landmarks a reconstruction must have lest it be discarded.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_minimum_robust_landmarks")
    )]
    pub minimum_robust_landmarks: usize,
    /// The the minimum number of robust observations for a landmark it to be considered robust enough for optimization
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_robust_minimum_observations")
    )]
    pub robust_minimum_observations: usize,
    /// The threshold of all observations in a landmark relative to another landmark to merge the two.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_merge_maximum_cosine_distance")
    )]
    pub merge_maximum_cosine_distance: f64,
    /// The minimum tripple product absolute value of three observations of a landmark for it to be
    /// considered robust enough for optimization
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_robust_observation_incidence_minimum_parallelepiped_volume")
    )]
    pub robust_observation_incidence_minimum_parallelepiped_volume: f64,
    /// The threshold used for single-view sample consensus
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_single_view_consensus_threshold")
    )]
    pub single_view_consensus_threshold: f64,
    /// The maximum number of matches to use in optimization.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_single_view_optimization_num_matches")
    )]
    pub single_view_optimization_num_matches: usize,
    /// The maximum iterations to run single-view optimization and filtering
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_single_view_filter_loop_iterations")
    )]
    pub single_view_filter_loop_iterations: usize,
    /// The maximum iterations to optimize one view.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_single_view_patience")
    )]
    pub single_view_patience: usize,
    /// The rotation magnitude at which single view optimization is terminated.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_single_view_epsilon")
    )]
    pub single_view_epsilon: f64,
    /// The threshold of mean cosine distance standard deviation that terminates single-view optimization.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_single_view_std_dev_threshold")
    )]
    pub single_view_std_dev_threshold: f64,
    /// The minimum number of 3d landmarks required for single-view registration.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_single_view_minimum_landmarks")
    )]
    pub single_view_minimum_landmarks: usize,
    /// The ratio of good matches to total matches to consider a single-view match successful.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_single_view_inlier_minimum_threshold")
    )]
    pub single_view_inlier_minimum_threshold: f64,
    /// The minimum number of robust landmarks matched to consider a single-view match successful.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_single_view_minimum_robust_landmarks")
    )]
    pub single_view_minimum_robust_landmarks: usize,
    /// The difference between the first and second best match above which a match is allowed for frame registration
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_single_view_match_better_by")
    )]
    pub single_view_match_better_by: u32,
    /// The threshold used for two-view sample consensus
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_two_view_consensus_threshold")
    )]
    pub two_view_consensus_threshold: f64,
    /// The minimum ratio of good matches to total matches to consider a two-view match successful.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_two_view_inlier_minimum_threshold")
    )]
    pub two_view_inlier_minimum_threshold: f64,
    /// The minimum number of matches to consider a two-view match successful.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_two_view_minimum_robust_matches")
    )]
    pub two_view_minimum_robust_matches: usize,
    /// The difference between the first and second best match above which a match is allowed for initialization
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_two_view_match_better_by")
    )]
    pub two_view_match_better_by: u32,
    /// The maximum number of matches to use for two-view optimization.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_two_view_optimization_maximum_matches")
    )]
    pub two_view_optimization_maximum_matches: usize,
    /// The maximum iterations to optimize two views.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_two_view_patience")
    )]
    pub two_view_patience: usize,
    /// The standard deviation threshold to successfully terminate two-view optimization.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_two_view_std_dev_threshold")
    )]
    pub two_view_std_dev_threshold: f64,
    /// The maximum iterations to optimize three views.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_three_view_patience")
    )]
    pub three_view_patience: usize,
    /// The minimum number of relative scales required for three-view optimization.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_three_view_minimum_relative_scales")
    )]
    pub three_view_minimum_relative_scales: usize,
    /// The threshold of mean cosine distance standard deviation that terminates three-view optimization.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_three_view_std_dev_threshold")
    )]
    pub three_view_std_dev_threshold: f64,
    /// The maximum iterations to run three-view optimization and filtering
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_three_view_filter_loop_iterations")
    )]
    pub three_view_filter_loop_iterations: usize,
    /// The minimum number of common matches that satisfy robustness criteria needed for initialization.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_three_view_optimization_landmarks")
    )]
    pub three_view_optimization_landmarks: usize,
    /// The ratio of good matches to total matches to consider a three-view match successful.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_three_view_inlier_ratio_threshold")
    )]
    pub three_view_inlier_ratio_threshold: f64,
    /// The minimum number of robust matches needed for creating a reconstruction to succeed.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_three_view_minimum_robust_matches")
    )]
    pub three_view_minimum_robust_matches: usize,
    /// The number of iterations to run bundle adjust, filtering, and merging.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_reconstruction_optimization_iterations")
    )]
    pub reconstruction_optimization_iterations: usize,
    /// The number of most similar frames to attempt to match when tracking.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_tracking_similar_frames")
    )]
    pub tracking_similar_frames: usize,
    /// The number of frames in the future or past in the feed the frame comes from
    /// that it must be distant by for it to be included in the similar frames.
    /// Similar frames from other feeds are always permitted to match.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_tracking_similar_frame_recent_threshold")
    )]
    pub tracking_similar_frame_recent_threshold: usize,
    /// The number of frames to search when trying to find similar frames for tracking.
    /// Increasing this number only increases the search size and does not
    /// cause more than `tracking_similar_frames` frames to be matched.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_tracking_similar_frame_search_num")
    )]
    pub tracking_similar_frame_search_num: usize,
    /// The number of most recent frames to attempt to match when tracking.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_tracking_recent_frames")
    )]
    pub tracking_recent_frames: usize,
    /// The maximum number of iterations to optimize three-view constraints.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_optimization_three_view_constraint_patience")
    )]
    pub optimization_three_view_constraint_patience: usize,
    /// The maximum number of three-view constraints per view.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_optimization_maximum_three_view_constraints")
    )]
    pub optimization_maximum_three_view_constraints: usize,
    /// The minimum number of three-view constraints that must be added to a new view.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_optimization_minimum_new_constraints")
    )]
    pub optimization_minimum_new_constraints: usize,
    /// The threshold of mean cosine distance standard deviation that terminates three-view optimization
    /// during graph optimization.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_optimization_std_dev_threshold")
    )]
    pub optimization_std_dev_threshold: f64,
    /// The number of optimization iterations.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_optimization_iterations")
    )]
    pub optimization_iterations: usize,
    /// The number of landmarks to use for optimization
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_optimization_landmarks")
    )]
    pub optimization_landmarks: usize,
    /// The minimum landmarks required for a three-view pose to be considered robust
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_optimization_robust_covisibility_minimum_landmarks")
    )]
    pub optimization_robust_covisibility_minimum_landmarks: usize,
    /// The multiplier that controls the convergence rate for optimization
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_optimization_convergence_rate")
    )]
    pub optimization_convergence_rate: f64,
}

impl Default for VSlamSettings {
    fn default() -> Self {
        Self {
            akaze_threshold: default_akaze_threshold(),
            maximum_cosine_distance: default_maximum_cosine_distance(),
            maximum_sine_distance: default_maximum_sine_distance(),
            minimum_robust_landmarks: default_minimum_robust_landmarks(),
            robust_minimum_observations: default_robust_minimum_observations(),
            merge_maximum_cosine_distance: default_merge_maximum_cosine_distance(),
            robust_observation_incidence_minimum_parallelepiped_volume:
                default_robust_observation_incidence_minimum_parallelepiped_volume(),
            single_view_consensus_threshold: default_single_view_consensus_threshold(),
            single_view_optimization_num_matches: default_single_view_optimization_num_matches(),
            single_view_filter_loop_iterations: default_single_view_filter_loop_iterations(),
            single_view_patience: default_single_view_patience(),
            single_view_epsilon: default_single_view_epsilon(),
            single_view_std_dev_threshold: default_single_view_std_dev_threshold(),
            single_view_minimum_landmarks: default_single_view_minimum_landmarks(),
            single_view_inlier_minimum_threshold: default_single_view_inlier_minimum_threshold(),
            single_view_minimum_robust_landmarks: default_single_view_minimum_robust_landmarks(),
            single_view_match_better_by: default_single_view_match_better_by(),
            two_view_consensus_threshold: default_two_view_consensus_threshold(),
            two_view_inlier_minimum_threshold: default_two_view_inlier_minimum_threshold(),
            two_view_minimum_robust_matches: default_two_view_minimum_robust_matches(),
            two_view_match_better_by: default_two_view_match_better_by(),
            two_view_std_dev_threshold: default_two_view_std_dev_threshold(),
            two_view_optimization_maximum_matches: default_two_view_optimization_maximum_matches(),
            two_view_patience: default_two_view_patience(),
            three_view_patience: default_three_view_patience(),
            three_view_std_dev_threshold: default_three_view_std_dev_threshold(),
            three_view_filter_loop_iterations: default_three_view_filter_loop_iterations(),
            three_view_optimization_landmarks: default_three_view_optimization_landmarks(),
            three_view_inlier_ratio_threshold: default_three_view_inlier_ratio_threshold(),
            three_view_minimum_robust_matches: default_three_view_minimum_robust_matches(),
            three_view_minimum_relative_scales: default_three_view_minimum_relative_scales(),
            reconstruction_optimization_iterations: default_reconstruction_optimization_iterations(
            ),
            tracking_similar_frames: default_tracking_similar_frames(),
            tracking_similar_frame_recent_threshold:
                default_tracking_similar_frame_recent_threshold(),
            tracking_similar_frame_search_num: default_tracking_similar_frame_search_num(),
            tracking_recent_frames: default_tracking_recent_frames(),
            optimization_three_view_constraint_patience:
                default_optimization_three_view_constraint_patience(),
            optimization_maximum_three_view_constraints:
                default_optimization_maximum_three_view_constraints(),
            optimization_minimum_new_constraints: default_optimization_minimum_new_constraints(),
            optimization_std_dev_threshold: default_optimization_std_dev_threshold(),
            optimization_iterations: default_optimization_iterations(),
            optimization_landmarks: default_optimization_landmarks(),
            optimization_robust_covisibility_minimum_landmarks:
                default_optimization_robust_covisibility_minimum_landmarks(),
            optimization_convergence_rate: default_optimization_convergence_rate(),
        }
    }
}

fn default_akaze_threshold() -> f64 {
    1e-4
}

fn default_maximum_cosine_distance() -> f64 {
    5e-7
}

fn default_maximum_sine_distance() -> f64 {
    5e-4
}

fn default_minimum_robust_landmarks() -> usize {
    1 << 8
}

fn default_robust_minimum_observations() -> usize {
    3
}

fn default_merge_maximum_cosine_distance() -> f64 {
    default_maximum_cosine_distance()
}

fn default_robust_observation_incidence_minimum_parallelepiped_volume() -> f64 {
    1e-4
}

fn default_single_view_consensus_threshold() -> f64 {
    // Cosine distance
    5e-7
}

fn default_single_view_optimization_num_matches() -> usize {
    1 << 11
}

fn default_single_view_filter_loop_iterations() -> usize {
    5
}

fn default_single_view_patience() -> usize {
    100000
}

fn default_single_view_epsilon() -> f64 {
    1e-6
}

fn default_single_view_std_dev_threshold() -> f64 {
    1e-16
}

fn default_single_view_minimum_landmarks() -> usize {
    1 << 5
}

fn default_single_view_inlier_minimum_threshold() -> f64 {
    0.0
}

fn default_single_view_minimum_robust_landmarks() -> usize {
    1 << 6
}

fn default_single_view_match_better_by() -> u32 {
    2
}

fn default_two_view_consensus_threshold() -> f64 {
    // Sine distance
    5e-4
}

fn default_two_view_inlier_minimum_threshold() -> f64 {
    0.0
}

fn default_two_view_minimum_robust_matches() -> usize {
    1 << 8
}

fn default_two_view_match_better_by() -> u32 {
    2
}

fn default_two_view_std_dev_threshold() -> f64 {
    1e-16
}

fn default_two_view_optimization_maximum_matches() -> usize {
    1 << 9
}

fn default_two_view_patience() -> usize {
    1 << 12
}

fn default_three_view_patience() -> usize {
    1 << 16
}

fn default_three_view_minimum_relative_scales() -> usize {
    1 << 4
}

fn default_three_view_std_dev_threshold() -> f64 {
    1e-16
}

fn default_three_view_filter_loop_iterations() -> usize {
    1 << 3
}

fn default_three_view_optimization_landmarks() -> usize {
    1 << 10
}

fn default_three_view_inlier_ratio_threshold() -> f64 {
    0.0
}

fn default_three_view_minimum_robust_matches() -> usize {
    1 << 8
}

fn default_reconstruction_optimization_iterations() -> usize {
    1
}

fn default_tracking_similar_frames() -> usize {
    1 << 5
}

fn default_tracking_similar_frame_recent_threshold() -> usize {
    0
}

fn default_tracking_similar_frame_search_num() -> usize {
    1 << 9
}

fn default_tracking_recent_frames() -> usize {
    0
}

fn default_optimization_three_view_constraint_patience() -> usize {
    20000
}

fn default_optimization_maximum_three_view_constraints() -> usize {
    1 << 4
}

fn default_optimization_minimum_new_constraints() -> usize {
    2
}

fn default_optimization_std_dev_threshold() -> f64 {
    1e-16
}

fn default_optimization_iterations() -> usize {
    1 << 10
}

fn default_optimization_landmarks() -> usize {
    1 << 6
}

fn default_optimization_robust_covisibility_minimum_landmarks() -> usize {
    1 << 4
}

fn default_optimization_convergence_rate() -> f64 {
    0.001
}
