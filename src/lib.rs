#![no_std]
extern crate alloc;

const BASIS_XXX: usize = 0;
const BASIS_XXY: usize = 1;
const BASIS_XYY: usize = 2;
const BASIS_YYY: usize = 3;
const BASIS_XXZ: usize = 4;
const BASIS_XYZ: usize = 5;
const BASIS_YYZ: usize = 6;
const BASIS_XZZ: usize = 7;
const BASIS_YZZ: usize = 8;
const BASIS_ZZZ: usize = 9;
const BASIS_XX: usize = 10;
const BASIS_XY: usize = 11;
const BASIS_YY: usize = 12;
const BASIS_XZ: usize = 13;
const BASIS_YZ: usize = 14;
const BASIS_ZZ: usize = 15;
const BASIS_X: usize = 16;
const BASIS_Y: usize = 17;
const BASIS_Z: usize = 18;
const BASIS_1: usize = 19;

use alloc::{vec, vec::Vec};
use cv_core::nalgebra::{
    self,
    dimension::{Dynamic, U1, U10, U20, U3, U4, U5, U6, U9},
    Matrix3, MatrixMN, SymmetricEigen, Vector4, VectorN,
};
use cv_core::{EssentialMatrix, NormalizedKeyPoint};

const EIGEN_CONVERGENCE: f32 = 1e-9;
const EIGEN_ITERATIONS: usize = 100;

type Input = MatrixMN<f32, U3, U5>;
type BasisVec = VectorN<f32, U20>;

fn encode_epipolar_equation(x1: &Input, x2: &Input) -> MatrixMN<f32, U5, U9> {
    let mut a: MatrixMN<f32, U5, U9> = nalgebra::zero();
    for i in 0..5 {
        for j in 0..3 {
            let v = x2[(j, i)] * x1.column(i).transpose();
            a.fixed_slice_mut::<U1, U3>(i, 3 * j).copy_from(&v);
        }
    }
    a
}

fn five_points_nullspace_basis(x1: &Input, x2: &Input) -> Option<MatrixMN<f32, U9, U4>> {
    let epipolar_constraint = encode_epipolar_equation(x1, x2);
    let ee = epipolar_constraint.transpose() * epipolar_constraint;
    ee.try_symmetric_eigen(EIGEN_CONVERGENCE, EIGEN_ITERATIONS)
        .map(|m| m.eigenvectors.fixed_columns::<U4>(0).into_owned())
}

fn o1(a: Vector4<f32>, b: Vector4<f32>) -> BasisVec {
    let mut res = BasisVec::zeros();
    res[BASIS_XX] = a.x * b.x;
    res[BASIS_XY] = a.x * b.y + a.y * b.x;
    res[BASIS_XZ] = a.x * b.z + a.z * b.x;
    res[BASIS_YY] = a.y * b.y;
    res[BASIS_YZ] = a.y * b.z + a.z * b.y;
    res[BASIS_ZZ] = a.z * b.z;
    res[BASIS_X] = a.x * b.w + a.w * b.x;
    res[BASIS_Y] = a.y * b.w + a.w * b.y;
    res[BASIS_Z] = a.z * b.w + a.w * b.z;
    res[BASIS_1] = a.w * b.w;
    res
}

fn o2(a: &BasisVec, b: &BasisVec) -> BasisVec {
    let mut res = BasisVec::zeros();
    res[BASIS_XXX] = a[BASIS_XX] * b[BASIS_X];
    res[BASIS_XXY] = a[BASIS_XX] * b[BASIS_Y] + a[BASIS_XY] * b[BASIS_X];
    res[BASIS_XXZ] = a[BASIS_XX] * b[BASIS_Z] + a[BASIS_XZ] * b[BASIS_X];
    res[BASIS_XYY] = a[BASIS_XY] * b[BASIS_Y] + a[BASIS_YY] * b[BASIS_X];
    res[BASIS_XYZ] = a[BASIS_XY] * b[BASIS_Z] + a[BASIS_YZ] * b[BASIS_X] + a[BASIS_XZ] * b[BASIS_Y];
    res[BASIS_XZZ] = a[BASIS_XZ] * b[BASIS_Z] + a[BASIS_ZZ] * b[BASIS_X];
    res[BASIS_YYY] = a[BASIS_YY] * b[BASIS_Y];
    res[BASIS_YYZ] = a[BASIS_YY] * b[BASIS_Z] + a[BASIS_YZ] * b[BASIS_Y];
    res[BASIS_YZZ] = a[BASIS_YZ] * b[BASIS_Z] + a[BASIS_ZZ] * b[BASIS_Y];
    res[BASIS_ZZZ] = a[BASIS_ZZ] * b[BASIS_Z];
    res[BASIS_XX] = a[BASIS_XX] * b[BASIS_1] + a[BASIS_X] * b[BASIS_X];
    res[BASIS_XY] = a[BASIS_XY] * b[BASIS_1] + a[BASIS_X] * b[BASIS_Y] + a[BASIS_Y] * b[BASIS_X];
    res[BASIS_XZ] = a[BASIS_XZ] * b[BASIS_1] + a[BASIS_X] * b[BASIS_Z] + a[BASIS_Z] * b[BASIS_X];
    res[BASIS_YY] = a[BASIS_YY] * b[BASIS_1] + a[BASIS_Y] * b[BASIS_Y];
    res[BASIS_YZ] = a[BASIS_YZ] * b[BASIS_1] + a[BASIS_Y] * b[BASIS_Z] + a[BASIS_Z] * b[BASIS_Y];
    res[BASIS_ZZ] = a[BASIS_ZZ] * b[BASIS_1] + a[BASIS_Z] * b[BASIS_Z];
    res[BASIS_X] = a[BASIS_X] * b[BASIS_1] + a[BASIS_1] * b[BASIS_X];
    res[BASIS_Y] = a[BASIS_Y] * b[BASIS_1] + a[BASIS_1] * b[BASIS_Y];
    res[BASIS_Z] = a[BASIS_Z] * b[BASIS_1] + a[BASIS_1] * b[BASIS_Z];
    res[BASIS_1] = a[BASIS_1] * b[BASIS_1];
    return res;
}
