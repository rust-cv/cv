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
//! The crate is designed to work with `#![no_std]`, even without an allocator. `libm` is used
//! (indirectly through [`num-traits`]) for all math algorithms that aren't present in `std`. Any
//! code that doesn't need to be shared across all CV crates should not belong in this repository.
//! If there is a good reason to put code that some crates may need into `cv-core`, it should be
//! gated behind a feature.
//!
//! ## Triangulation
//!
//! Several of the traits with in `cv-core`, such as [`TriangulatorObservations`], must perform a process
//! called [triangulation](https://en.wikipedia.org/wiki/Triangulation). In computer vision, this problem
//! occurs quite often, as we often have some of the following data:
//!
//! * [The pose of a camera](WorldToCamera)
//! * [The relative pose of a camera](CameraToCamera)
//! * [A bearing direction at which lies a feature](nalgebra::UnitVector3)
//!
//! We have to take this data and produce a 3d point. Cameras have an optical center which all bearings protrude from.
//! This is often refered to as the focal point in a standard camera, but in computer vision the term optical center
//! is prefered, as it is a generalized concept. What typically happens in triangulation is that we have (at least)
//! two optical centers and a bearing (direction) out of each of those optical centers approximately pointing towards
//! the 3d point. In an ideal world, these bearings would point exactly at the point and triangulation would be achieved
//! simply by solving the equation for the point of intersection. Unfortunately, the real world throws a wrench at us, as
//! the bearings wont actually intersect since they are based on noisy data. This is what causes us to need different
//! triangulation algorithms, which deal with the error in different ways and have different characteristics.
//!
//! Here is an example where we have two pinhole cameras A and B. The `@` are used to show the
//! [virtual image plane](https://en.wikipedia.org/wiki/Pinhole_camera_model). The virtual image plane can be thought
//! of as a surface in front of the camera through which the light passes through from the point to the optical center `O`.
//! The points `a` and `b` are normalized image coordinates which describe the position on the virtual image plane which
//! the light passed through from the point to the optical center on cameras `A` and `B` respectively. We know the
//! exact pose (position and orientation) of each of these two cameras, and we also know the normalized image coordinates,
//! which we can use to compute a bearing. We are trying to solve for the point `p` which would cause the ray of light to
//! pass through points `a` and `b` followed by `O`.
//!
//! - `p` the point we are trying to triangulate
//! - `a` the normalized keypoint on camera A
//! - `b` the normalized keypoint on camera B
//! - `O` the optical center of a camera
//! - `@` the virtual image plane
//!
//! ```text
//!                        @
//!                        @
//!               p--------b--------O
//!              /         @
//!             /          @
//!            /           @
//!           /            @
//!   @@@@@@@a@@@@@
//!         /
//!        /
//!       /
//!      O
//! ```

#![no_std]

mod camera;
mod keypoint;
mod matches;
mod point;
mod pose;
mod so3;
mod triangulation;

pub use camera::*;
pub use keypoint::*;
pub use matches::*;
pub use nalgebra;
pub use point::*;
pub use pose::*;
pub use sample_consensus;
pub use so3::*;
pub use triangulation::*;
