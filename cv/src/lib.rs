//! # `cv`
//!
//! Batteries-included pure-Rust computer vision crate
//!
//! This crate should only be used for documentation/reference and for quickly creating and
//! running a computer vision sample/routine. It is useful for tutorials and also for
//! experts who want to run something once as a script. It also stores all of the
//! things useful for computer vision in the Rust ecosystem in one place for
//! discoverability. If you are making a production application, import the dependencies
//! from this crate individually so that you don't have an explosive number of dependencies.
//! Although not recommended, you can also disable default features on this crate and
//! enable specific features on this crate just to get the functionality you want.
//!
//! All of the basic computer vision types/dependencies are included in the root of the crate.
//! Modules are created to store algorithms and data structures which may or may not be used.
//! Almost all of the things in these modules come from optional libraries.
//! These modules comprise the core functionality required to perform computer vision tasks.
//!
//! ## Modules
//! * [`camera`] - camera models to convert image coordinates into bearings (and back)
//! * [`consensus`] - finding the best estimated model from noisy data
//! * [`geom`] - computational geometry algorithms used in computer vision
//! * [`estimate`] - estimation of models from data
//! * [`feature`] - feature extraction and description
//! * [`image`] - image opening and processing/manipulation
//! * [`knn`] - searching for nearest neighbors in small or large datasets
//! * [`optimize`] - optimizing models to fit data
//! * [`mvg`] - multiple-view geometry (visual odometry, SfM, vSLAM)
//! * [`video`] - video opening and camera capture
//! * [`vis`] - visualization

#![no_std]

pub use cv_core::{sample_consensus::*, *};

#[cfg(feature = "space")]
pub use space::Metric;

#[cfg(feature = "bitarray")]
pub use bitarray;

/// Camera models (see [`video`] for camera capture)
pub mod camera {
    /// The pinhole camera model
    #[cfg(feature = "cv-pinhole")]
    pub use cv_pinhole as pinhole;
}

/// Consensus algorithms (RANSAC)
pub mod consensus {
    #[cfg(feature = "arrsac")]
    pub use arrsac::Arrsac;
}

/// Computational geometry
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
    #[cfg(feature = "nister-stewenius")]
    pub use nister_stewenius::NisterStewenius;
}

/// Feature detection and description
pub mod feature {
    /// A robust and fast feature detector
    #[cfg(feature = "akaze")]
    pub mod akaze {
        pub use akaze::*;
    }
}

/// Image opening and processing/manipulation
pub mod image {
    /// Re-export of [`image`] to open and save images
    #[cfg(feature = "image")]
    #[allow(clippy::module_inception)]
    pub mod image {
        pub use image::*;
    }

    /// Re-export of [`imageproc`] crate for image manipulation routines
    #[cfg(feature = "imageproc")]
    pub mod imageproc {
        pub use imageproc::*;
    }

    /// Re-export of [`ndarray-vision`] for image manipulation routines
    #[cfg(feature = "ndarray-vision")]
    pub mod ndarray_vision {
        pub use ndarray_vision::*;
    }
}

/// Algorithms for performing k-NN searches
pub mod knn {
    /// Re-export of [`hgg`] crate, an approximate nearest neighbor search map
    #[cfg(feature = "hgg")]
    pub mod hgg {
        pub use hgg::*;
    }

    /// Re-export of [`hnsw`] crate, an approximate nearest neighbor index search data structure
    #[cfg(feature = "hnsw")]
    pub mod hnsw {
        pub use hnsw::*;
    }

    #[cfg(all(feature = "space", feature = "alloc"))]
    pub use space::{KnnInsert, KnnMap, KnnPoints, LinearKnn};

    #[cfg(feature = "space")]
    pub use space::{Knn, Metric, Neighbor};
}

/// Optimization algorithms
pub mod optimize {
    /// Levenberg-Marquardt
    #[cfg(feature = "levenberg-marquardt")]
    pub mod lm {
        pub use levenberg_marquardt::*;
    }
}

/// Multiple-view geometry (visual odometry, SfM, vSLAM)
pub mod mvg {
    #[cfg(feature = "cv-sfm")]
    pub use cv_sfm as sfm;
}

/// Video and camera capture
pub mod video {
    /// Re-export of [`eye`] crate, used for capturing camera input
    #[cfg(feature = "eye")]
    pub mod eye {
        pub use eye::*;
    }
}

/// Visualization utilities
pub mod vis {}
