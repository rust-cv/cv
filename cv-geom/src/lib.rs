//! This crate contains computational geometry algorithms for [Rust CV](https://github.com/rust-cv/).
//!
//! ## Triangulation
//!
//! In this problem we know the relative pose of cameras and the [`Bearing`] of the same feature
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

use cv_core::nalgebra::{zero, Matrix3x4, Matrix4, RowVector4};
use cv_core::{
    Bearing, CameraPoint, CameraToCamera, Pose, TriangulatorObservations, TriangulatorRelative,
    WorldPoint, WorldToCamera,
};

/// This solves triangulation problems by simply minimizing the squared reprojection error of all observances.
///
/// This is a quick triangulator to execute and is fairly robust.
///
/// ```
/// use cv_core::nalgebra::{Vector3, Point3, Rotation3, Unit};
/// use cv_core::{TriangulatorRelative, CameraToCamera, CameraPoint, Pose, Projective};
/// use cv_geom::MinSquaresTriangulator;
///
/// let point = CameraPoint::from_point(Point3::new(0.3, 0.1, 2.0));
/// let pose = CameraToCamera::from_parts(Vector3::new(0.1, 0.1, 0.1), Rotation3::new(Vector3::new(0.1, 0.1, 0.1)));
/// let bearing_a = point.bearing();
/// let bearing_b = pose.transform(point).bearing();
/// let triangulated = MinSquaresTriangulator::new().triangulate_relative(pose, bearing_a, bearing_b).unwrap();
/// let distance = (point.point().unwrap().coords - triangulated.point().unwrap().coords).norm();
/// assert!(distance < 1e-6);
/// ```
#[derive(Copy, Clone, Debug)]
pub struct MinSquaresTriangulator {
    epsilon: f64,
    max_iterations: usize,
}

impl MinSquaresTriangulator {
    /// Creates a `MinSquaresTriangulator` with default values.
    ///
    /// Same as calling [`Default::default`].
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

impl Default for MinSquaresTriangulator {
    fn default() -> Self {
        Self {
            epsilon: 1e-9,
            max_iterations: 100,
        }
    }
}

impl TriangulatorObservations for MinSquaresTriangulator {
    fn triangulate_observations<B: Bearing>(
        &self,
        pairs: impl IntoIterator<Item = (WorldToCamera, B)>,
    ) -> Option<WorldPoint> {
        let mut a: Matrix4<f64> = zero();
        let mut count = 0;

        for (pose, bearing) in pairs {
            count += 1;
            // Get the normalized bearing.
            let bearing = bearing.bearing().into_inner();
            // Get the pose as a 3x4 matrix.
            let rot = pose.0.rotation.matrix();
            let trans = pose.0.translation.vector;
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

        if count < 2 {
            return None;
        }

        let se = a.try_symmetric_eigen(self.epsilon, self.max_iterations)?;

        se.eigenvalues
            .iter()
            .enumerate()
            .min_by_key(|&(_, &n)| float_ord::FloatOrd(n))
            .map(|(ix, _)| se.eigenvectors.column(ix).into_owned())
            .map(|v| if v.w.is_sign_negative() { -v } else { v })
            .map(Into::into)
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
    /// Same as calling [`Default::default`].
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
        relative_pose: CameraToCamera,
        a: A,
        b: B,
    ) -> Option<CameraPoint> {
        let pose = relative_pose.homogeneous();
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
            .map(|v| if v.w.is_sign_negative() { -v } else { v })
            .map(Into::into)
    }
}
