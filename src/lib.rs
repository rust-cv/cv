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
    Matrix3, MatrixMN, Vector4, VectorN,
};
use cv_core::{EssentialMatrix, NormalizedKeyPoint};

type Input = MatrixMN<f32, U3, U5>;

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
