//! This module contains functions to perform various geometric algorithms.
//!
//! ## Triangulation of a point with a given camera transformation
//!
//! In this problem we have a [`RelativeCameraPose`](crate::RelativeCameraPose) and two [`Bearing`].
//! We want to find the point of intersection from the two cameras in camera A's space.
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
//!
//! //! Solutions to this problem:
//!
//! * [`make_one_pose_dlt_triangulator`]
//!
//! ## Translation along a bearing given one prior depth
//!
//! This problem consumes a direction to translate along, a `from` [`CameraPoint`],
//! and a `to` [`Bearing`] coordinate.
//!
//! - `t` the `translation` bearing vector
//! - `a` the `from` point
//! - `b` the `to` epipolar point
//! - `O` the optical center
//! - `@` the virtual image plane
//!
//! ```text
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
//!
//! ## Triangulation from variable number of camera poses
//!
//! In this problem you have several camera poses and bearing pairs, where each bearing is an observation
//! on the image produced from the corresponding camera pose. The desired output is a triangulated point.
//!
//! * [`triangulate_least_square_reprojection_error`]

#![no_std]

use cv_core::nalgebra::{zero, Matrix3x4, Matrix4, RowVector4, Vector3};
use cv_core::{
    Bearing, CameraPoint, CameraPose, RelativeCameraPose, TriangulatorObservances,
    TriangulatorProject, TriangulatorRelative, WorldPoint,
};

/// This solves the translation along a bearing triangulation assuming that there is
/// a perfect intersection.
#[derive(Copy, Clone, Debug)]
pub struct BearingIntersectionTriangulator;

impl TriangulatorProject for BearingIntersectionTriangulator {
    fn triangulate_project<B: Bearing>(
        &self,
        from: CameraPoint,
        onto: B,
        translation: Vector3<f64>,
    ) -> Option<f64> {
        let from = from.0.coords;
        let to = onto.bearing_unnormalized();

        let hv = to.cross(&-from);
        let h = hv.norm();
        let kv = to.cross(&translation);
        let k = kv.norm();

        let l = h / k;

        Some(if hv.dot(&kv) > 0.0 { l } else { -l })
    }
}

/// This solves the translation along a bearing triangulation by minimizing the reprojection error.
#[derive(Copy, Clone, Debug)]
pub struct BearingMinimizeReprojectionErrorTriangulator;

impl TriangulatorProject for BearingMinimizeReprojectionErrorTriangulator {
    fn triangulate_project<B: Bearing>(
        &self,
        from: CameraPoint,
        onto: B,
        translation: Vector3<f64>,
    ) -> Option<f64> {
        let a = onto.bearing_unnormalized();
        let b = from;
        let t = translation;
        Some((a.y * b.x - a.x * b.y) / (a.x * t.y - a.y * t.x))
    }
}

/// This solves triangulation problems by simply minimizing the squared reprojection error of all observances.
///
/// This is a quick triangulator to execute and is fairly robust.
#[derive(Copy, Clone, Debug)]
pub struct MinimalSquareReprojectionErrorTriangulator {
    epsilon: f64,
    max_iterations: usize,
}

impl MinimalSquareReprojectionErrorTriangulator {
    /// Creates a `MinimalSquareReprojectionErrorTriangulator` with default values.
    ///
    /// Same as [`MinimalSquareReprojectionErrorTriangulator::default`].
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the epsilon used in the symmetric eigen solver.
    ///
    /// Default is `1e-9`.
    pub fn epsilon(self, epsilon: f64) -> Self {
        Self { epsilon, ..self }
    }

    /// Set the maximum number of iterations for the symmetric eigen solver.
    ///
    /// Default is `100`.
    pub fn max_iterations(self, max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..self
        }
    }
}

impl Default for MinimalSquareReprojectionErrorTriangulator {
    fn default() -> Self {
        Self {
            epsilon: 1e-9,
            max_iterations: 100,
        }
    }
}

impl TriangulatorObservances for MinimalSquareReprojectionErrorTriangulator {
    fn triangulate_observances<B: Bearing>(
        &self,
        pairs: impl IntoIterator<Item = (CameraPose, B)>,
    ) -> Option<WorldPoint> {
        let mut a: Matrix4<f64> = zero();

        for (pose, bearing) in pairs {
            // Get the normalized bearing.
            let bearing = bearing.bearing().into_inner();
            // Get the pose as a 3x4 matrix.
            let rot = pose.rotation.matrix();
            let trans = pose.translation.vector;
            let pose = Matrix3x4::<f64>::from_columns(&[
                rot.column(0),
                rot.column(1),
                rot.column(2),
                trans.column(0),
            ]);
            // Set up the least squares problem.
            let term = pose - bearing * bearing.transpose() * pose;
            a += term.transpose() * term;
        }

        let se = a.try_symmetric_eigen(self.epsilon, self.max_iterations)?;

        se.eigenvalues
            .iter()
            .enumerate()
            .min_by_key(|&(_, &n)| float_ord::FloatOrd(n))
            .map(|(ix, _)| se.eigenvectors.column(ix).into_owned())
            .map(|h| (h.xyz() / h.w).into())
            .map(WorldPoint)
    }
}

/// Based on algorithm 12 from "Multiple View Geometry in Computer Vision, Second Edition"
#[derive(Copy, Clone, Debug)]
pub struct RelativeDltTriangulator {
    epsilon: f64,
    max_iterations: usize,
}

impl RelativeDltTriangulator {
    /// Creates a `RelativeDltTriangulator` with default values.
    ///
    /// Same as [`RelativeDltTriangulator::default`].
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the epsilon used in the SVD solver.
    ///
    /// Default is `1e-9`.
    pub fn epsilon(self, epsilon: f64) -> Self {
        Self { epsilon, ..self }
    }

    /// Set the maximum number of iterations for the SVD solver.
    ///
    /// Default is `100`.
    pub fn max_iterations(self, max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..self
        }
    }
}

impl Default for RelativeDltTriangulator {
    fn default() -> Self {
        Self {
            epsilon: 1e-9,
            max_iterations: 100,
        }
    }
}

impl TriangulatorRelative for RelativeDltTriangulator {
    fn triangulate_relative<A: Bearing, B: Bearing>(
        &self,
        relative_pose: RelativeCameraPose,
        a: A,
        b: B,
    ) -> Option<CameraPoint> {
        let pose = relative_pose.to_homogeneous();
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

        let svd = design.try_svd(false, true, self.epsilon, self.max_iterations)?;

        // Extract the null-space vector from V* corresponding to the smallest
        // singular value and then normalize it back from heterogeneous coordinates.
        svd.singular_values
            .iter()
            .enumerate()
            .min_by_key(|&(_, &n)| float_ord::FloatOrd(n))
            .map(|(ix, _)| svd.v_t.unwrap().row(ix).transpose().into_owned())
            .map(|h| (h.xyz() / h.w).into())
            .map(CameraPoint)
    }
}
