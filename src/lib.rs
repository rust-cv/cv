//! # Rust CV Core
//!
//! This library provides common abstractions and types for computer vision (CV) in Rust.
//! All the crates in the rust-cv ecosystem that have or depend on CV types depend on this crate.
//! This includes things like camera model traits, bearings, poses, keypoints, etc. The crate is designed to
//! be very small so that it adds negligable build time. It pulls in some dependencies
//! that will probably be brought in by writing computer vision code normally.
//! The core concept is that all CV crates can work together with each other by using the
//! abstractions and types specified in this crate.
//!
//! The crate is designed to work with `#![no_std]`, even without an allocator. [`libm`] is used
//! for all math algorithms that aren't present in `std`. Any code that doesn't need to be shared
//! across all CV crates should not belong in this repository. If there is a good reason to put
//! code that some crates may need into `cv-core`, it should be gated behind a feature.

#![no_std]
mod camera;
pub mod geom;
mod keypoint;
mod matches;
mod pose;
mod so3;
mod world;

pub use camera::*;
pub use keypoint::*;
pub use matches::*;
pub use nalgebra;
pub use pose::*;
pub use sample_consensus;
pub use so3::*;
pub use world::*;
