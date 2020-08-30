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
//! * [`geom`] - computational geometry algorithms used in computer vision
//! * [`estimate`] - estimation of models from data
//! * [`feature`] - feature extraction and description
//! * [`knn`] - searching for nearest neighbors in small or large datasets
//! * [`optimize`] - optimizing models to best fit the data
//! * [`vis`] - visualization utilities

#![no_std]

pub use cv_core::sample_consensus::*;
pub use cv_core::*;

#[cfg(feature = "space")]
pub use space::MetricPoint;

#[cfg(feature = "bitarray")]
pub use bitarray::BitArray;

/// Camera models
pub mod camera {
    /// The pinhole camera model
    #[cfg(feature = "cv-pinhole")]
    pub use cv_pinhole as pinhole;
}

/// Consensus algorithms
pub mod consensus {
    #[cfg(feature = "arrsac")]
    pub use arrsac::Arrsac;
}

/// Computational geometry algorithms
pub mod geom {
    #[cfg(feature = "cv-geom")]
    pub use cv_geom::*;
}

/// Estimation algorithms
pub mod estimate {
    #[cfg(feature = "eight-point")]
    pub use eight_point::EightPoint;
    #[cfg(feature = "lambda-twist")]
    pub use lambda_twist::LambdaTwist;
}

/// Feature detection and description algorithms
pub mod feature {
    /// A robust and fast feature detector
    #[cfg(feature = "akaze")]
    pub mod akaze {
        pub use akaze::*;
    }
}

/// Algorithms for performing k-NN searches
pub mod knn {
    /// An approximate nearest neighbor index search data structure
    #[cfg(feature = "hnsw")]
    pub mod hnsw {
        pub use hnsw::*;
    }

    #[cfg(feature = "space")]
    pub use space::linear_knn;
}

/// Optimization algorithms
pub mod optimize {
    /// Levenberg-Marquardt
    #[cfg(feature = "levenberg-marquardt")]
    pub mod lm {
        pub use levenberg_marquardt::*;
    }
}

/// Visualization utilities
pub mod vis {
    #[cfg(feature = "imgshow")]
    pub use imgshow::imgshow;
}
