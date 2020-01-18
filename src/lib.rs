// Copyright (C) 2013 The Regents of the University of California (Regents).
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
//
//     * Neither the name of The Regents or University of California nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Please contact the author of this library if you have any questions.
// Author: Chris Sweeney (cmsweeney@cs.ucsb.edu)
//
// Translated to Rust by Geordon Worley <vadixidav@gmail.com>

#![no_std]
extern crate alloc;

use alloc::{vec, vec::Vec};
use cv_core::nalgebra::{
    self,
    dimension::{U10, U20, U3, U4, U5, U9},
    Matrix3, MatrixMN, Vector2, Vector4, VectorN,
};

/// Multiply two degree one polynomials of variables x, y, z.
/// E.g. p1 = a[0]x + a[1]y + a[2]z + a[3]
/// Output order: x^2 xy y^2 xz yz z^2 x y z 1 (GrevLex)
fn multiply_deg_one_poly(a: Vector4<f32>, b: Vector4<f32>) -> VectorN<f32, U10> {
    VectorN::<f32, U10>::from_iterator(
        [
            // x^2
            a.x * b.x,
            // xy
            a.x * b.y + a.y * b.x,
            // y^2
            a.y * b.y,
            // xz
            a.x * b.z + a.z * b.x,
            // yz
            a.y * b.z + a.z * b.y,
            // z^2
            a.z * b.z,
            // x
            a.x * b.w + a.w * b.x,
            // y
            a.y * b.w + a.w * b.y,
            // z
            a.z * b.w + a.w * b.z,
            // 1
            a.w * b.w,
        ]
        .iter()
        .copied(),
    )
}

// Multiply a 2 deg poly (in x, y, z) and a one deg poly in GrevLex order.
// x^3 x^2y xy^2 y^3 x^2z xyz y^2z xz^2 yz^2 z^3 x^2 xy y^2 xz yz z^2 x y z 1
fn multiply_deg_two_deg_one_poly(a: VectorN<f32, U10>, b: Vector4<f32>) -> VectorN<f32, U20> {
    VectorN::<f32, U20>::from_iterator(
        [
            // x^3
            a[0] * b.x,
            // x^2y
            a[0] * b.y + a[1] * b.x,
            // xy^2
            a[1] * b.y + a[2] * b.x,
            // y^3
            a[2] * b.y,
            // x^2z
            a[0] * b.z + a[3] * b.x,
            // xyz
            a[1] * b.z + a[3] * b.y + a[4] * b.x,
            // y^2z
            a[2] * b.z + a[4] * b.y,
            // xz^2
            a[3] * b.z + a[5] * b.x,
            // yz^2
            a[4] * b.z + a[5] * b.y,
            // z^3
            a[5] * b.z,
            // x^2
            a[0] * b.w + a[6] * b.x,
            // xy
            a[1] * b.w + a[6] * b.y + a[7] * b.x,
            // y^2
            a[2] * b.w + a[7] * b.y,
            // xz
            a[3] * b.w + a[6] * b.z + a[8] * b.x,
            // yz
            a[4] * b.w + a[7] * b.z + a[8] * b.y,
            // z^2
            a[5] * b.w + a[8] * b.z,
            // x
            a[6] * b.w + a[9] * b.x,
            // y
            a[7] * b.w + a[9] * b.y,
            // z
            a[8] * b.w + a[9] * b.z,
            // 1
            a[9] * b.w,
        ]
        .iter()
        .copied(),
    )
}

fn get_determinant_constraint(null_space: [[Vector4<f32>; 3]; 3]) -> VectorN<f32, U20> {
    // Singularity constraint.
    multiply_deg_two_deg_one_poly(
        multiply_deg_one_poly(null_space[0][1], null_space[1][2])
            - multiply_deg_one_poly(null_space[0][2], null_space[1][1]),
        null_space[2][0],
    ) + multiply_deg_two_deg_one_poly(
        multiply_deg_one_poly(null_space[0][2], null_space[1][0])
            - multiply_deg_one_poly(null_space[0][0], null_space[1][2]),
        null_space[2][1],
    ) + multiply_deg_two_deg_one_poly(
        multiply_deg_one_poly(null_space[0][0], null_space[1][1])
            - multiply_deg_one_poly(null_space[0][1], null_space[1][0]),
        null_space[2][2],
    )
}

// Shorthand for multiplying the Essential matrix with its transpose.
fn essential_self_transpose(
    null_space: [[Vector4<f32>; 3]; 3],
    i: usize,
    j: usize,
) -> VectorN<f32, U10> {
    multiply_deg_one_poly(null_space[i][0], null_space[j][0])
        + multiply_deg_one_poly(null_space[i][1], null_space[j][1])
        + multiply_deg_one_poly(null_space[i][2], null_space[j][2])
}

// Builds the trace constraint: EEtE - 1/2 trace(EEt)E = 0
fn get_trace_constraint(null_space: [[Vector4<f32>; 3]; 3]) -> MatrixMN<f32, U9, U20> {
    // Comput EEt.
    let mut eet = [[VectorN::<f32, U10>::zeros(); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            eet[i][j] = 2.0 * essential_self_transpose(null_space, i, j);
        }
    }

    // Compute the trace.
    let trace = eet[0][0] + eet[1][1] + eet[2][2];

    let mut trace_constraint = MatrixMN::<f32, U9, U20>::zeros();

    // Multiply EEt with E.
    for (ix, mut row) in trace_constraint.row_iter_mut().enumerate() {
        let i = ix / 3;
        let j = ix % 3;
        let src = multiply_deg_two_deg_one_poly(eet[i][0], null_space[0][j])
            + multiply_deg_two_deg_one_poly(eet[i][1], null_space[1][j])
            + multiply_deg_two_deg_one_poly(eet[i][2], null_space[2][j])
            - 0.5 * multiply_deg_two_deg_one_poly(trace, null_space[i][j]);
        row.copy_from(&src.transpose());
    }

    trace_constraint
}

fn build_constraint_matrix(null_space: [[Vector4<f32>; 3]; 3]) -> MatrixMN<f32, U10, U20> {
    let mut constraint_matrix: MatrixMN<f32, U10, U20> = nalgebra::zero();
    constraint_matrix
        .fixed_rows_mut::<U9>(0)
        .copy_from(&get_trace_constraint(null_space));
    constraint_matrix
        .row_mut(9)
        .copy_from(&get_determinant_constraint(null_space).transpose());
    constraint_matrix
}

// Implementation of Nister from "An Efficient Solution to the Five-Point Relative Pose Problem".
pub fn five_point_relative_pose(
    max_iter: usize,
    image1_points: &[Vector2<f32>],
    image2_points: &[Vector2<f32>],
) -> Vec<Matrix3<f32>> {
    assert_eq!(image1_points.len(), image2_points.len());
    assert!(
        image1_points.len() >= 5,
        "You must supply at least 5 correspondences for the 5 point essential matrix algorithm"
    );

    // Step 1. Create the 5x9 matrix containing epipolar constraints.
    //   Essential matrix is a linear combination of the 4 vectors spanning the
    //   null space of this matrix.
    let mut epipolar_constraint: MatrixMN<f32, U5, U9> = nalgebra::zero();
    for (i, points) in image1_points.iter().zip(image2_points).enumerate() {
        // Fill matrix with the epipolar constraint from q'_t*E*q = 0. Where q is
        // from the first image, and q' is from the second.
        epipolar_constraint.row_mut(i).copy_from_slice(&[
            points.1.x * points.0.x,
            points.1.y * points.0.x,
            points.0.x,
            points.1.x * points.0.y,
            points.1.y * points.0.y,
            points.0.y,
            points.1.x,
            points.1.y,
            1.0,
        ]);
    }

    // This always performs SVD because I was not sure how to get FullPivLU
    // to compute the null space for me. nalgebra does not include the routine.
    let v_t = if let Some(svd) =
        (epipolar_constraint.transpose() * epipolar_constraint).try_svd(false, true, 1e-6, max_iter)
    {
        svd.v_t.unwrap()
    } else {
        return vec![];
    };
    // TODO: Make sure this doesn't need to be transposed.
    let null_space = v_t.fixed_rows::<U4>(5);

    let null_space_matrix = [
        [
            null_space.column(0).into(),
            null_space.column(3).into(),
            null_space.column(6).into(),
        ],
        [
            null_space.column(1).into(),
            null_space.column(4).into(),
            null_space.column(7).into(),
        ],
        [
            null_space.column(2).into(),
            null_space.column(5).into(),
            null_space.column(8).into(),
        ],
    ];

    // Step 2. Expansion of the epipolar constraints on the determinant and trace.
    let constraint_matrix = build_constraint_matrix(null_space_matrix);

    // Step 3. Eliminate part of the matrix to isolate polynomials in z.
    let c_lu =
        nalgebra::FullPivLU::new(constraint_matrix.fixed_slice::<U10, U10>(0, 0).into_owned());
    let eliminated_matrix = if let Some(solved) = c_lu.solve(
        &constraint_matrix
            .fixed_slice::<U10, U10>(0, 10)
            .into_owned(),
    ) {
        solved
    } else {
        return vec![];
    };

    let mut action_matrix: MatrixMN<f32, U10, U10> = nalgebra::zero();
    action_matrix
        .fixed_slice_mut::<U3, U10>(0, 0)
        .copy_from(&eliminated_matrix.fixed_slice::<U3, U10>(0, 0));

    action_matrix
        .row_mut(3)
        .copy_from(&eliminated_matrix.row(4));
    action_matrix
        .row_mut(4)
        .copy_from(&eliminated_matrix.row(5));
    action_matrix
        .row_mut(5)
        .copy_from(&eliminated_matrix.row(7));
    action_matrix[(6, 0)] = -1.0;
    action_matrix[(7, 1)] = -1.0;
    action_matrix[(8, 3)] = -1.0;
    action_matrix[(9, 6)] = -1.0;

    let nalgebra::SymmetricEigen { eigenvectors, .. } =
        if let Some(m) = nalgebra::SymmetricEigen::try_new(action_matrix, 1e-6, max_iter) {
            m
        } else {
            return vec![];
        };

    // Now that we have x, y, and z we need to substitute them back into the null
    // space to get a valid essential matrix solution.
    let null_space_transpose = null_space.transpose();
    eigenvectors
        .fixed_rows::<U4>(5)
        .column_iter()
        .map(|eigenvector| {
            Matrix3::from_iterator((null_space_transpose * eigenvector).iter().copied())
        })
        .collect()
}
