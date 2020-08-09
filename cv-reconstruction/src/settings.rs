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
    /// The threshold distance below which a match is allowed
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_match_threshold")
    )]
    pub match_threshold: usize,
    /// The threshold used for sample consensus
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_consensus_threshold")
    )]
    pub consensus_threshold: f64,
    /// The number of points to use in optimization of matches
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_optimization_points")
    )]
    pub optimization_points: usize,
    /// The minimum cosine distance of a landmark for it to be considered robust enough for optimization
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_incidence_minimum_cosine_distance")
    )]
    pub incidence_minimum_cosine_distance: f64,
    /// The maximum cosine distance of an observation for it to be considered robust enough for optimization
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_robust_maximum_cosine_distance")
    )]
    pub robust_maximum_cosine_distance: f64,
    /// The the minimum number of robust observations for a landmark it to be considered robust enough for optimization
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_robust_minimum_observations")
    )]
    pub robust_minimum_observations: usize,
    /// The cutoff for the loss function
    #[cfg_attr(feature = "serde-serialize", serde(default = "default_loss_cutoff"))]
    pub loss_cutoff: f64,
    /// The maximum cosine distance permitted in a valid match
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_cosine_distance_threshold")
    )]
    pub cosine_distance_threshold: f64,
    /// The threshold of all observations in a landmark relative to another landmark to merge the two.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_merge_cosine_distance_threshold")
    )]
    pub merge_cosine_distance_threshold: f64,
    /// The maximum iterations to optimize one view.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_single_view_patience")
    )]
    pub single_view_patience: usize,
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
    /// The cosine distance threshold during initialization.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_two_view_cosine_distance_threshold")
    )]
    pub two_view_cosine_distance_threshold: f64,
    /// The maximum iterations to optimize two views.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_two_view_patience")
    )]
    pub two_view_patience: usize,
    /// The threshold of mean cosine distance standard deviation that terminates two-view optimization.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_two_view_std_dev_threshold")
    )]
    pub two_view_std_dev_threshold: f64,
    /// The maximum iterations to run two-view optimization and filtering
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_two_view_filter_loop_iterations")
    )]
    pub two_view_filter_loop_iterations: usize,
    /// The maximum number of landmarks to use for pose estimation during tracking.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_track_landmarks")
    )]
    pub track_landmarks: usize,
    /// The maximum iterations to optimize many views.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_many_view_patience")
    )]
    pub many_view_patience: usize,
    /// The threshold of mean cosine distance standard deviation that terminates many-view optimization.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_many_view_std_dev_threshold")
    )]
    pub many_view_std_dev_threshold: f64,
    /// The number of landmarks to use in bundle adjust.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_many_view_landmarks")
    )]
    pub many_view_landmarks: usize,
    /// The number of iterations to run bundle adjust, filtering, and merging.
    #[cfg_attr(
        feature = "serde-serialize",
        serde(default = "default_reconstruction_optimization_iterations")
    )]
    pub reconstruction_optimization_iterations: usize,
}

impl Default for VSlamSettings {
    fn default() -> Self {
        Self {
            akaze_threshold: default_akaze_threshold(),
            match_threshold: default_match_threshold(),
            consensus_threshold: default_consensus_threshold(),
            optimization_points: default_optimization_points(),
            incidence_minimum_cosine_distance: default_incidence_minimum_cosine_distance(),
            robust_maximum_cosine_distance: default_robust_maximum_cosine_distance(),
            robust_minimum_observations: default_robust_minimum_observations(),
            loss_cutoff: default_loss_cutoff(),
            cosine_distance_threshold: default_cosine_distance_threshold(),
            merge_cosine_distance_threshold: default_merge_cosine_distance_threshold(),
            single_view_patience: default_single_view_patience(),
            single_view_std_dev_threshold: default_single_view_std_dev_threshold(),
            single_view_minimum_landmarks: default_single_view_minimum_landmarks(),
            two_view_cosine_distance_threshold: default_two_view_cosine_distance_threshold(),
            two_view_patience: default_two_view_patience(),
            two_view_std_dev_threshold: default_two_view_std_dev_threshold(),
            two_view_filter_loop_iterations: default_two_view_filter_loop_iterations(),
            track_landmarks: default_track_landmarks(),
            many_view_patience: default_many_view_patience(),
            many_view_std_dev_threshold: default_many_view_std_dev_threshold(),
            many_view_landmarks: default_many_view_landmarks(),
            reconstruction_optimization_iterations: default_reconstruction_optimization_iterations(
            ),
        }
    }
}

fn default_akaze_threshold() -> f64 {
    0.00001
}

fn default_match_threshold() -> usize {
    64
}

fn default_consensus_threshold() -> f64 {
    0.001
}

fn default_optimization_points() -> usize {
    8192
}

fn default_incidence_minimum_cosine_distance() -> f64 {
    0.0005
}

fn default_robust_maximum_cosine_distance() -> f64 {
    0.0000001
}

fn default_robust_minimum_observations() -> usize {
    3
}

fn default_loss_cutoff() -> f64 {
    0.00002
}

fn default_cosine_distance_threshold() -> f64 {
    0.001
}

fn default_merge_cosine_distance_threshold() -> f64 {
    0.0000005
}

fn default_single_view_patience() -> usize {
    8000
}

fn default_single_view_std_dev_threshold() -> f64 {
    0.00000000001
}

fn default_single_view_minimum_landmarks() -> usize {
    32
}

fn default_two_view_cosine_distance_threshold() -> f64 {
    0.001
}

fn default_two_view_patience() -> usize {
    8000
}

fn default_two_view_std_dev_threshold() -> f64 {
    0.00000001
}

fn default_two_view_filter_loop_iterations() -> usize {
    3
}

fn default_track_landmarks() -> usize {
    4096
}

fn default_many_view_patience() -> usize {
    8000
}

fn default_many_view_std_dev_threshold() -> f64 {
    0.000000000001
}

fn default_many_view_landmarks() -> usize {
    32768
}

fn default_reconstruction_optimization_iterations() -> usize {
    1
}
