//! # `cv`
//!
//! Batteries-included pure-Rust computer vision crate
//!
//! All of the basic computer vision types are included in the root of the crate.
//! Modules are created to store algorithms and data structures which may or may not be used.
//! Almost all of the things in these modules come from optional libraries.
//! These modules comprise the core functionality required to perform computer vision tasks.
//!
//! Some crates are re-exported to ensure that you can use the same version of the crate
//! that `cv` is using.
//!
//! ## Modules
//! * [`camera`] - camera models to convert image coordinates into bearings (and back)
//! * [`consensus`] - finding the best estimated model from noisy data
//! * [`estimate`] - estimation of models from data
//! * [`feature`] - feature extraction and description
//! * [`knn`] - searching for nearest neighbors in small or large datasets
//! * [`optimize`] - optimizing models to best fit the data

#![no_std]

pub use cv_core::{
    nalgebra,
    sample_consensus::{Consensus, Estimator, Model, MultiConsensus},
    Bearing, CameraModel, CameraPoint, CameraPose, EssentialMatrix, FeatureMatch,
    FeatureWorldMatch, ImagePoint, KeyPoint, RelativeCameraPose, UnscaledRelativeCameraPose,
    WorldPoint, WorldPose,
};

pub use space::{self, MetricPoint};

/// Camera models
pub mod camera {
    /// The pinhole camera model
    #[cfg(feature = "pinhole")]
    pub use cv_core::pinhole;
}

/// Consensus algorithms
pub mod consensus {
    #[cfg(feature = "arrsac")]
    pub use arrsac::{Arrsac, Config as ArrsacConfig};
}

/// Estimation algorithms
pub mod estimate {
    #[cfg(feature = "eight-point")]
    pub use eight_point::EightPoint;
}

/// Feature detection and description algorithms
pub mod feature {
    #[cfg(feature = "akaze")]
    pub use akaze;
}

/// Algorithms for performing k-NN searches
pub mod knn {
    #[cfg(feature = "hnsw")]
    pub use hnsw;

    pub use space::linear_knn;
}

/// Optimization algorithms
pub mod optimize {
    #[cfg(feature = "levenberg-marquardt")]
    pub use levenberg_marquardt::{optimize as lm, Config as LMConfig};
}
