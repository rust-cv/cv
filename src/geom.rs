//! This module contains functions to perform various geometric algorithms.
//!
//! ## Translation along a bearing given one prior depth
//!
//! This problem consumes a direction to translate along, a `from` [`CameraPoint`],
//! and a `to` [`NormalizedKeyPoint`] coordinate.
//!
//! - `t` the `translation` bearing vector
//! - `a` the `from` point
//! - `b` the `to` epipolar point
//! - `O` the optical center
//! - `@` the virtual image plane
//!
//! ```no_build,no_run
//!      t<---a
//!          /
//!         /
//! @@@b@@@/@@@@@
//!    |  /
//!    | /
//!    |/
//!    O
//! ```
//!
//! The `from` coordinate is the relative 3d coordinate in camera space before translation.
//!
//! The `to` coordinate is just a normalized keypoint that we wish to find the optimal translation
//! to reproject as close as possible to.
//!
//! The `translation` is a vector which will be scaled (multiplied) by the return value to
//! get the actual 3d translation to move from `from` to `to` in 3d space.
//!
//! Solutions to this problem:
//!
//! * [`triangulate_bearing_intersection`]
//! * [`triangulate_bearing_reproject`]
//!
//! It is recommended to use [`triangulate_bearing_reproject`], as it is incredibly cheap to compute.

use crate::{CameraPoint, NormalizedKeyPoint};
use nalgebra::Vector3;

/// This solves the translation along a bearing triangulation assuming that there is
/// a perfect intersection.
pub fn triangulate_bearing_intersection(
    bearing: Vector3<f64>,
    from: CameraPoint,
    to: NormalizedKeyPoint,
) -> f64 {
    let from = from.0.coords;
    let to = to.bearing_unnormalized();

    let hv = to.cross(&-from);
    let h = hv.norm();
    let kv = to.cross(&bearing);
    let k = kv.norm();

    let l = h / k;

    if hv.dot(&kv) > 0.0 {
        l
    } else {
        -l
    }
}

/// This solves the translation along a bearing triangulation by minimizing the reprojection error.
pub fn triangulate_bearing_reproject(
    bearing: Vector3<f64>,
    from: CameraPoint,
    to: NormalizedKeyPoint,
) -> f64 {
    let a = to;
    let b = from;
    let t = bearing;
    (a.y * b.x - a.x * b.y) / (a.x * t.y - a.y * t.x)
}
