//! This crate contains computational geometry algorithms for [Rust CV](https://github.com/rust-cv/).
//!
//! ## Triangulation
//!
//! In this problem we know the relative pose of cameras and the bearing of the same feature
//! observed in each camera frame. We want to find the point of intersection from all cameras.
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

pub mod epipolar;
pub mod triangulation;
