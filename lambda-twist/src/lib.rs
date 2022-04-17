//! This package provides functions to solve camera pose estimation
//! given a set of three 3D points and their corresponding pixel coordinates.
//!
//! This problem is generally known as ["Perspective-n-Point" (PnP)][pnp].
//! We focus on the P3P case (n = 3) which is the
//! minimal amount of points required to solve the problem with a finite number of solutions.
//! There are many approaches to solve P3P. In this package, we have implemented:
//!
//!  - Lambda Twist: An Accurate Fast Robust Perspective Three Point (P3P) Solver.
//!    Mikael Persson, Klas Nordberg. ECCV 2018. ([paper][lambda-twist])
//!
//! [pnp]: https://en.wikipedia.org/wiki/Perspective-n-Point
//! [lambda-twist]: http://openaccess.thecvf.com/content_ECCV_2018/html/Mikael_Persson_Lambda_Twist_An_ECCV_2018_paper.html
//!
//! Implementation based on
//! "Lambda Twist: An Accurate Fast Robust Perspective Three Point (P3P) Solver"
//! Persson, M. and Nordberg, K. ECCV 2018.
//! Reference implementation available on the [author github repository][lambda-twist-github].
//!
//! [lambda-twist-github]: https://github.com/midjji/lambdatwist-p3p
#![no_std]
#![warn(missing_docs)]

use arrayvec::ArrayVec;
use num_traits::Float;

// Copyright (c) 2018 Michael Persson
// Adapted to openMVG by Romain Janvier and Pierre Moulon
// Adapted to Rust by Matthieu Pizenberg

// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use cv_core::{
    nalgebra::{Matrix3, Rotation3, Vector3},
    sample_consensus::Estimator,
    FeatureWorldMatch, Pose, Projective, WorldToCamera,
};

type Mat3 = Matrix3<f64>;
type Vec3 = Vector3<f64>;

/// The core P3P algorithm named Lambda Twist. Estimates up to 4 potential [`WorldToCamera`] poses.
///
/// Returns 0 to 4 potential solutions to the equation:
///
/// ![Formula](https://chart.googleapis.com/chart?cht=tx&chl=\lambda_i%20\%20y_i%20=%20R%20x_i%20%2B%20t,\quad%20i%20\in%20\{1,%202,%203\})
///
/// - ![x_i](https://chart.googleapis.com/chart?cht=tx&chl=x_i) are the 3D point world coordinates.
/// - ![y_i](https://chart.googleapis.com/chart?cht=tx&chl=y_i) are the image coordinates ![Formula](https://chart.googleapis.com/chart?cht=tx&chl=y_i%20\%20\sim%20\%20(u_i,\%20v_i,\%201))
///   (also sometimes called "bearing vectors").
/// - ![lambda_i](https://chart.googleapis.com/chart?cht=tx&chl=lambda_i) are the signed distances from the camera.
///
/// This struct is marked as `#[non_exhaustive]` to allow the backwards-compatible addition of new fields.
#[derive(Copy, Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct LambdaTwist {
    /// This determines the number of iterations that Gauss-Newton is ran on the poses to optimize them.
    /// The paper notes that it rarely improves after two iterations. The original implementation uses 5 iterations.
    pub gauss_newton_iterations: usize,
    /// This determines the number of iterations to spend converging on a proper rotation matrix.
    pub rotation_convergence_iterations: usize,
    /// The epsilon on which convergence is considered achieved for converging on a proper rotation.
    pub rotation_convergence_epsilon: f64,
}

impl LambdaTwist {
    /// Creates a default lambda twist with 5 iterations of Gauss-Newton.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the [`LambdaTwist::gauss_newton_iterations`].
    #[must_use]
    pub fn gauss_newton_iterations(self, gauss_newton_iterations: usize) -> Self {
        Self {
            gauss_newton_iterations,
            ..self
        }
    }

    /// Sets the [`LambdaTwist::rotation_convergence_iterations`].
    #[must_use]
    pub fn rotation_convergence_iterations(self, rotation_convergence_iterations: usize) -> Self {
        Self {
            rotation_convergence_iterations,
            ..self
        }
    }

    /// Sets the [`LambdaTwist::rotation_convergence_epsilon`].
    #[must_use]
    pub fn rotation_convergence_epsilon(self, rotation_convergence_epsilon: f64) -> Self {
        Self {
            rotation_convergence_epsilon,
            ..self
        }
    }

    /// Compute the pose of a camera using three 3D-to-2D correspondences.
    /// Implementation of the paper
    /// "Lambda Twist: An Accurate Fast Robust Perspective Three Point (P3P) Solver".
    /// Persson, M. and Nordberg, K. ECCV 2018.
    ///
    /// `let poses = compute_poses_nordberg(world_3d_points, bearing_vectors);`
    ///
    /// The 3x3 matrix `world_3d_points` contains one 3D point per column.
    /// The 3x3 matrix `bearing_vectors` contains one homogeneous image coordinate per column.
    fn compute_poses_nordberg(
        &self,
        samples: [FeatureWorldMatch; 3],
    ) -> ArrayVec<WorldToCamera, 4> {
        // Extraction of 3D points vectors
        let to_wp = |&FeatureWorldMatch(_, point)| point.point();
        let wps = [
            if let Some(p) = to_wp(&samples[0]) {
                p
            } else {
                return ArrayVec::new();
            },
            if let Some(p) = to_wp(&samples[1]) {
                p
            } else {
                return ArrayVec::new();
            },
            if let Some(p) = to_wp(&samples[2]) {
                p
            } else {
                return ArrayVec::new();
            },
        ];
        let to_bearing = |&FeatureWorldMatch(bearing, _): &FeatureWorldMatch| bearing;
        let bearings = [
            to_bearing(&samples[0]),
            to_bearing(&samples[1]),
            to_bearing(&samples[2]),
        ];

        // Compute vectors between 3D points.
        let d12 = wps[0] - wps[1];
        let d13 = wps[0] - wps[2];
        let d23 = wps[1] - wps[2];
        let d12xd13 = d12.cross(&d13);

        // "a12" is the squared distance between 3D points 1 and 2.
        let a12 = d12.norm_squared();
        let a13 = d13.norm_squared();
        let a23 = d23.norm_squared();

        // "c31" is the cosine between bearing vectors 3 and 1.
        let c12 = bearings[0].dot(&bearings[1]);
        let c23 = bearings[1].dot(&bearings[2]);
        let c31 = bearings[2].dot(&bearings[0]);
        let blob = c12 * c23 * c31 - 1.0;

        // "s31" is the sine between bearing vectors 3 and 1.
        let s12_sqr = 1.0 - c12 * c12;
        let s23_sqr = 1.0 - c23 * c23;
        let s31_sqr = 1.0 - c31 * c31;

        // Other useful constants.
        let b12 = -2.0 * c12;
        let b13 = -2.0 * c31;
        let b23 = -2.0 * c23;

        // "p[0-3]" here are the four coefficients of the cubic polynomial.
        // They are refered to as "c[0-3]" in equation (10) of the paper.
        let p3 = a13 * (a23 * s31_sqr - a13 * s23_sqr);
        let p2 = 2.0 * blob * a23 * a13
            + a13 * (2.0 * a12 + a13) * s23_sqr
            + a23 * (a23 - a12) * s31_sqr;
        let p1 = a23 * (a13 - a23) * s12_sqr
            - a12 * a12 * s23_sqr
            - 2.0 * a12 * (blob * a23 + a13 * s23_sqr);
        let p0 = a12 * (a12 * s23_sqr - a23 * s12_sqr);

        // Get sharpest real root of above.
        let g = cube_root(p2 / p3, p1 / p3, p0 / p3);

        // Build the matrix called D0 in the paper.
        let d0_00 = a23 * (1.0 - g);
        let d0_01 = -(a23 * c12);
        let d0_02 = a23 * c31 * g;
        let d0_11 = a23 - a12 + a13 * g;
        let d0_12 = -c23 * (a13 * g - a12);
        let d0_22 = g * (a13 - a23) - a12;
        #[rustfmt::skip]
    let d0_mat = Mat3::new(
        d0_00, d0_01, d0_02,
        d0_01, d0_11, d0_12,
        d0_02, d0_12, d0_22,
    );

        // Get sorted eigenvectors and eigenvalues of the singular matrix D0.
        let (eig_vectors, eig_values) = eigen_decomposition_singular(d0_mat);

        // Initialize the possible depths triplets for the three image points.
        // There might be between 0 and 4 possible solutions.
        let mut lambdas: ArrayVec<Vec3, 4> = ArrayVec::new();

        // Solve the four possible solutions for the depths values.
        let eigen_ratio = (0.0_f64.max(-eig_values[1] / eig_values[0])).sqrt();

        // Helper closure to compute quadratic coefficients.
        // CF equation (15) in paper.
        let quadratic_coefficients = |ratio: f64| {
            let w2 = 1.0 / (ratio * eig_vectors.m12 - eig_vectors.m11);
            let w0 = w2 * (eig_vectors.m21 - ratio * eig_vectors.m22);
            let w1 = w2 * (eig_vectors.m31 - ratio * eig_vectors.m32);

            let a = 1.0 / ((a13 - a12) * w1 * w1 - a12 * b13 * w1 - a12);
            let b = a * (a13 * b12 * w1 - a12 * b13 * w0 - 2.0 * w0 * w1 * (a12 - a13));
            let c = a * ((a13 - a12) * w0 * w0 + a13 * b12 * w0 + a13);
            (w0, w1, b, c)
        };

        // Helper closure to estimate possible depths values.
        // CF equation (16) in paper.
        let possible_depths = |tau: f64, w0: f64, w1: f64| {
            let d = a23 / (tau * (b23 + tau) + 1.0);
            if d > 0.0 {
                let l2 = d.sqrt();
                let l3 = tau * l2;
                let l1 = w0 * l2 + w1 * l3;
                (true, l1, l2, l3)
            } else {
                (false, 0.0, 0.0, 0.0)
            }
        };

        // Helper closure pushing one potential solution.
        let mut push_solution = |tau: f64, w0: f64, w1: f64| {
            if tau > 0.0 {
                let (valid, l1, l2, l3) = possible_depths(tau, w0, w1);
                if valid && l1 >= 0.0 {
                    lambdas.push(Vec3::new(l1, l2, l3));
                }
            }
        };

        // Helper closure pushing two potential solutions
        // corresponding to a given eigen value ratio.
        let mut push_solutions_to_lambdas = |ratio: f64| {
            let (w0, w1, b, c) = quadratic_coefficients(ratio);
            if b * b - 4.0 * c >= 0.0 {
                let (_, tau1, tau2) = root2real(b, c);
                push_solution(tau1, w0, w1);
                push_solution(tau2, w0, w1);
            }
        };

        push_solutions_to_lambdas(eigen_ratio);
        push_solutions_to_lambdas(-eigen_ratio);

        // Recover the rotation R and translation t such that
        // lambda_i * y_i = R * x_i + t
        // TODO: replace by the quaternion version.
        // According to the author, it works slightly better.
        #[rustfmt::skip]
    let x_mat = Mat3::new(
        d12[0], d13[0], d12xd13[0],
        d12[1], d13[1], d12xd13[1],
        d12[2], d13[2], d12xd13[2],
    );
        let x_mat = if let Some(x_mat) = x_mat.try_inverse() {
            x_mat
        } else {
            return ArrayVec::new();
        };

        lambdas
            .iter()
            .map(|&lambda| {
                // Refine estimated depth values.
                let lambda_refined = gauss_newton_refine_lambda(
                    lambda,
                    self.gauss_newton_iterations,
                    a12,
                    a13,
                    a23,
                    b12,
                    b13,
                    b23,
                );

                let ry1 = lambda_refined[0] * *bearings[0];
                let ry2 = lambda_refined[1] * *bearings[1];
                let ry3 = lambda_refined[2] * *bearings[2];

                let yd1 = ry1 - ry2;
                let yd2 = ry1 - ry3;
                let yd1xd2 = yd1.cross(&yd2);

                #[rustfmt::skip]
            let y_mat = Mat3::new(
                yd1[0], yd2[0], yd1xd2[0],
                yd1[1], yd2[1], yd1xd2[1],
                yd1[2], yd2[2], yd1xd2[2],
            );

                let rot = y_mat * x_mat;
                (rot, ry1 - rot * wps[0].coords)
            })
            .map(|(rot, trans)| {
                WorldToCamera::from_parts(
                    trans,
                    Rotation3::from_matrix_eps(
                        &rot,
                        self.rotation_convergence_epsilon,
                        self.rotation_convergence_iterations,
                        Rotation3::identity(),
                    ),
                )
            })
            .collect()
    }
}

impl Default for LambdaTwist {
    fn default() -> Self {
        Self {
            gauss_newton_iterations: 5,
            rotation_convergence_iterations: 100,
            rotation_convergence_epsilon: 1e-12,
        }
    }
}

impl Estimator<FeatureWorldMatch> for LambdaTwist {
    type Model = WorldToCamera;
    type ModelIter = ArrayVec<WorldToCamera, 4>;
    const MIN_SAMPLES: usize = 3;
    fn estimate<I>(&self, mut data: I) -> Self::ModelIter
    where
        I: Iterator<Item = FeatureWorldMatch> + Clone,
    {
        self.compute_poses_nordberg([
            data.next()
                .expect("must provide 3 samples at minimum to LambdaTwist"),
            data.next()
                .expect("must provide 3 samples at minimum to LambdaTwist"),
            data.next()
                .expect("must provide 3 samples at minimum to LambdaTwist"),
        ])
    }
}

/// Refine a valid solution with a Gauss-Newton Solver.
/// `refined_lambda = gauss_newton_refine_lambda(lambda, a12, a13, a23, b12, b13, b23);`
/// lambda: Vec3, the solution to refine.
/// a12: f64, the squared distance between 3D point 1 and 3D point 2.
/// a13: f64, the squared distance between 3D point 1 and 3D point 3.
/// a23: f64, the squared distance between 3D point 2 and 3D point 3.
/// b12: f64, -2.0 * cosine of the angle between bearing vector 1 and bearing vector 2.
/// b13: f64, -2.0 * cosine of the angle between bearing vector 1 and bearing vector 3.
/// b23: f64, -2.0 * cosine of the angle between bearing vector 2 and bearing vector 3.
/// The paper note it rarely improve after two iterations. The original implementation use 5 iterations.
#[allow(clippy::similar_names)]
#[allow(clippy::too_many_arguments)]
fn gauss_newton_refine_lambda(
    lambda: Vec3,
    iterations: usize,
    a12: f64,
    a13: f64,
    a23: f64,
    b12: f64,
    b13: f64,
    b23: f64,
) -> Vec3 {
    let compute_residual = |l: &Vec3| {
        let l1 = l.x;
        let l2 = l.y;
        let l3 = l.z;
        let r1 = l1 * l1 + l2 * l2 + b12 * l1 * l2 - a12;
        let r2 = l1 * l1 + l3 * l3 + b13 * l1 * l3 - a13;
        let r3 = l2 * l2 + l3 * l3 + b23 * l2 * l3 - a23;
        (l1, l2, l3, Vec3::new(r1, r2, r3))
    };
    let (mut l1, mut l2, mut l3, mut res) = compute_residual(&lambda);
    for _ in 0..iterations {
        if l1_norm(res) < 1e-10 {
            break;
        }

        let dr1dl1 = 2.0 * l1 + b12 * l2;
        let dr1dl2 = 2.0 * l2 + b12 * l1;
        let dr2dl1 = 2.0 * l1 + b13 * l3;
        let dr2dl3 = 2.0 * l3 + b13 * l1;
        let dr3dl2 = 2.0 * l2 + b23 * l3;
        let dr3dl3 = 2.0 * l3 + b23 * l2;
        let det = 1.0 / (-dr1dl1 * dr2dl3 * dr3dl2 - dr1dl2 * dr2dl1 * dr3dl3);

        #[rustfmt::skip]
        let jacobian = Mat3::new(
            -dr2dl3 * dr3dl2, -dr1dl2 * dr3dl3,  dr1dl2 * dr2dl3,
            -dr2dl1 * dr3dl3,  dr1dl1 * dr3dl3, -dr1dl1 * dr2dl3,
             dr2dl1 * dr3dl2, -dr1dl1 * dr3dl2, -dr1dl2 * dr2dl1,
        );
        let lambda_new = Vec3::new(l1, l2, l3) - det * (jacobian * res);
        let (l1_new, l2_new, l3_new, res_new) = compute_residual(&lambda_new);
        if l1_norm(res_new) > l1_norm(res) {
            break;
        } else {
            l1 = l1_new;
            l2 = l2_new;
            l3 = l3_new;
            res = res_new;
        }
    }
    Vec3::new(l1, l2, l3)
}

/// Compute L1 norm of a vector.
/// L1 norm is the sum of magnitudes.
#[inline]
fn l1_norm(v: Vec3) -> f64 {
    v.x.abs() + v.y.abs() + v.z.abs()
}

/// Compute the real roots of "h(r) = r^2 + b*r + c = 0".
/// `let (roots_are_real, r1, r2) = root2real(b, c);`
fn root2real(b: f64, c: f64) -> (bool, f64, f64) {
    let discriminant = b * b - 4.0 * c;
    if discriminant < 0.0 {
        let root = 0.5 * b;
        (false, root, root)
    } else if b < 0.0 {
        let y = Float::sqrt(discriminant);
        (true, 0.5 * (-b + y), 0.5 * (-b - y))
    } else {
        let y = Float::sqrt(discriminant);
        (true, 2.0 * c / (-b + y), 2.0 * c / (-b - y))
    }
}

/// Compute a single root of the cubic polynomial equation
/// "h(r) = r^3 + b*r^2 + c*r + d = 0".
/// `let root = cubick(b, c, d);`
///
/// The return root is as stable as possible in the sense that it has as high
/// derivative as possible. The solution is found by simple Newton-Raphson iterations,
/// and the trick is to choose the initial solution r0 in a clever way.
///
/// The intial solution is found by considering 5 cases:
///
/// Cases I and II: h has no stationary points. In this case its derivative
/// is positive.  The inital solution to the NR-iteration is r0 here h has
/// minimal derivative.
///
/// Case III, IV, and V: has two stationary points, t1 < t2.
/// In this case, h has negative derivative between t1 and t2.
/// In these cases, we can make a second order approximation of h
/// around each of t1 and t2, and choose r0
/// as the leftmost or rightmost root of these approximations,
/// depending on whether zero, one, or both of h(t1) and h(t2) are > 0.
#[allow(clippy::many_single_char_names)]
fn cube_root(b: f64, c: f64, d: f64) -> f64 {
    // Choose an initial solution.
    let mut r0;
    // Not monotonic.
    if b * b >= 3.0 * c {
        // h has two stationary points, compute them.
        // double t1 = t - sqrt(diff);
        let v = (b * b - 3.0 * c).sqrt();
        let t1 = (-b - v) / 3.0;

        // Check if h(t1) > 0, in this case make a 2-order approx of h around t1.
        let mut k = ((t1 + b) * t1 + c) * t1 + d;

        if k > 0.0 {
            // Find leftmost root of 0.5 * (r0 - t1)^2 * (6 * t1 + 2 * b) + k = 0.
            r0 = t1 - (-k / (3.0 * t1 + b)).sqrt();
        } else {
            let t2 = (-b + v) / 3.0;
            k = ((t2 + b) * t2 + c) * t2 + d;
            // Find rightmost root of 0.5 * (r0 - t2)^2 * (6 * t2 + 2 * b) + k1 = 0.
            r0 = t2 + (-k / (3.0 * t2 + b)).sqrt();
        }
    } else {
        // r0 = 1.0 / cubick_inv(c/d, b/d, 1.0/d);
        r0 = -b / 3.0;
        if ((3.0 * r0 + 2.0 * b) * r0 + c).abs() < 1e-4 {
            r0 += 1.0;
        }
    }

    // Newton-Raphson iterations (hardcoded, perform at least 7 iterations, at most 50).
    // Break if position of root changes less than 1e-13.
    // According to the author, increasing it could lead to a better solution (more robust).
    for _ in 0..7 {
        let fx = ((r0 + b) * r0 + c) * r0 + d;
        let fpx = (3.0 * r0 + 2.0 * b) * r0 + c;
        r0 -= fx / fpx;
    }
    for _ in 0..43 {
        let fx = ((r0 + b) * r0 + c) * r0 + d;
        if fx.abs() > 1e-13 {
            let fpx = (3.0 * r0 + 2.0 * b) * r0 + c;
            r0 -= fx / fpx;
        } else {
            break;
        }
    }
    r0
}

/// Eigen decomposition of a singular matrix (which has a 0 eigen value).
/// `let (eigenvectors, eigenvalues) = eigen_decomposition_singular(x);`
fn eigen_decomposition_singular(x: Mat3) -> (Mat3, Vec3) {
    let mut eigenvalues = Vec3::zeros();
    #[rustfmt::skip]
    let mut v3 = Vec3::new(
        x[1] * x[5] - x[2] * x[4],
        x[2] * x[3] - x[5] * x[0],
        x[4] * x[0] - x[1] * x[3],
    );
    v3.normalize_mut();

    let x12_sqr = x.m12 * x.m12;
    let b = -x.m11 - x.m22 - x.m33;
    let c = -x12_sqr - x.m13 * x.m13 - x.m23 * x.m23 + x.m11 * (x.m22 + x.m33) + x.m22 * x.m33;
    let (_, mut e1, mut e2) = root2real(b, c);
    if e1.abs() < e2.abs() {
        core::mem::swap(&mut e1, &mut e2);
    }
    eigenvalues[0] = e1;
    eigenvalues[1] = e2;

    let mx0011 = -x.m11 * x.m22;
    let prec_0 = x.m12 * x.m23 - x.m13 * x.m22;
    let prec_1 = x.m12 * x.m13 - x.m11 * x.m23;

    let compute_eigen_vector = |e: f64| {
        let tmp = 1.0 / (e * (x.m11 + x.m22) + mx0011 - e * e + x12_sqr);
        let mut a1 = -(e * x.m13 + prec_0) * tmp;
        let mut a2 = -(e * x.m23 + prec_1) * tmp;
        let rnorm = 1.0 / (a1 * a1 + a2 * a2 + 1.0).sqrt();
        a1 *= rnorm;
        a2 *= rnorm;
        Vec3::new(a1, a2, rnorm)
    };
    let v1 = compute_eigen_vector(e1);
    let v2 = compute_eigen_vector(e2);

    #[rustfmt::skip]
    let eigenvectors = Mat3::new(
        v1[0], v2[0], v3[0],
        v1[1], v2[1], v3[1],
        v1[2], v2[2], v3[2],
    );

    (eigenvectors, eigenvalues)
}
