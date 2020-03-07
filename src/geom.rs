//! This module contains functions to perform various geometric algorithms.
//!
//! ## Triangulation of a point with a given camera transformation
//!
//! In this problem we have a [`RelativeCameraPose`] and two [`NormalizedKeyPoint`].
//! We want to find the point of intersection from the two cameras in camera A's space.
//!
//! - `p` the point we are trying to triangulate
//! - `a` the normalized keypoint on camera A
//! - `b` the normalized keypoint on camera B
//! - `O` the optical center of a camera
//! - `@` the virtual image plane
//!
//! ```no_build,no_run
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
//!
//! //! Solutions to this problem:
//!
//! * [`triangulate_dlt`]
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

use crate::{Bearing, CameraPoint, UnscaledRelativeCameraPose};
use nalgebra::{Matrix4, Point3, RowVector4, Vector3};

/// This solves the point triangulation problem using
/// Algorithm 12 from "Multiple View Geometry in Computer Vision".
///
/// It is considered the "optimal" triangulation and is best when dealing with noise.
pub fn make_one_pose_dlt_triangulator<B>(
    epsilon: f64,
    max_iterations: usize,
) -> impl Fn(UnscaledRelativeCameraPose, B, B) -> Option<Point3<f64>>
where
    B: Bearing,
{
    move |pose, a, b| {
        let pose = pose.to_homogeneous();
        let a = a.bearing_unnormalized();
        let b = b.bearing_unnormalized();
        let mut design = Matrix4::zeros();
        design
            .row_mut(0)
            .copy_from(&RowVector4::new(-a.z, 0.0, a.x, 0.0));
        design
            .row_mut(1)
            .copy_from(&RowVector4::new(0.0, -a.z, a.y, 0.0));
        design
            .row_mut(2)
            .copy_from(&(b.x * pose.row(2) - b.z * pose.row(0)));
        design
            .row_mut(3)
            .copy_from(&(b.y * pose.row(2) - b.z * pose.row(1)));

        let svd = design.try_svd(false, true, epsilon, max_iterations)?;

        // Extract the null-space vector from V* corresponding to the smallest
        // singular value and then normalize it back from heterogeneous coordinates.
        svd.singular_values
            .iter()
            .enumerate()
            .min_by_key(|&(_, &n)| float_ord::FloatOrd(n))
            .map(|(ix, _)| svd.v_t.unwrap().row(ix).transpose().into_owned())
            .map(|h| (h.xyz() / h.w).into())
    }
}

/// This solves the translation along a bearing triangulation assuming that there is
/// a perfect intersection.
pub fn triangulate_bearing_intersection<B>(
    bearing: Vector3<f64>,
    from: CameraPoint,
    to: B,
) -> Option<f64>
where
    B: Bearing,
{
    let from = from.0.coords;
    let to = to.bearing_unnormalized();

    let hv = to.cross(&-from);
    let h = hv.norm();
    let kv = to.cross(&bearing);
    let k = kv.norm();

    let l = h / k;

    Some(if hv.dot(&kv) > 0.0 { l } else { -l })
}

/// This solves the translation along a bearing triangulation by minimizing the reprojection error.
pub fn triangulate_bearing_reproject<B>(
    bearing: Vector3<f64>,
    from: CameraPoint,
    to: B,
) -> Option<f64>
where
    B: Bearing,
{
    let a = to.bearing_unnormalized();
    let b = from;
    let t = bearing;
    Some((a.y * b.x - a.x * b.y) / (a.x * t.y - a.y * t.x))
}
