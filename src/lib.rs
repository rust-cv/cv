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

use cv_core::nalgebra::{
    self,
    dimension::{U1, U10, U20, U3, U4, U5, U9},
    Matrix3, MatrixMN, MatrixN, Vector4, VectorN,
};
use cv_core::EssentialMatrix;

const EIGEN_CONVERGENCE: f32 = 1e-9;
const EIGEN_ITERATIONS: usize = 100;

type Input = MatrixMN<f32, U3, U5>;
type PolyBasisVec = VectorN<f32, U20>;
type NullspaceMat = MatrixMN<f32, U9, U4>;
type ConstraintMat = MatrixMN<f32, U10, U20>;
type Square10 = MatrixN<f32, U10>;

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

fn five_points_nullspace_basis(x1: &Input, x2: &Input) -> Option<NullspaceMat> {
    let epipolar_constraint = encode_epipolar_equation(x1, x2);
    let ee = epipolar_constraint.transpose() * epipolar_constraint;
    ee.try_symmetric_eigen(EIGEN_CONVERGENCE, EIGEN_ITERATIONS)
        .map(|m| m.eigenvectors.fixed_columns::<U4>(0).into_owned())
}

fn o1(a: Vector4<f32>, b: Vector4<f32>) -> PolyBasisVec {
    let mut res = PolyBasisVec::zeros();
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

fn o2(a: PolyBasisVec, b: Vector4<f32>) -> PolyBasisVec {
    let mut res = PolyBasisVec::zeros();
    res[BASIS_XXX] = a[BASIS_XX] * b.x;
    res[BASIS_XXY] = a[BASIS_XX] * b.y + a[BASIS_XY] * b.x;
    res[BASIS_XXZ] = a[BASIS_XX] * b.z + a[BASIS_XZ] * b.x;
    res[BASIS_XYY] = a[BASIS_XY] * b.y + a[BASIS_YY] * b.x;
    res[BASIS_XYZ] = a[BASIS_XY] * b.z + a[BASIS_YZ] * b.x + a[BASIS_XZ] * b.y;
    res[BASIS_XZZ] = a[BASIS_XZ] * b.z + a[BASIS_ZZ] * b.x;
    res[BASIS_YYY] = a[BASIS_YY] * b.y;
    res[BASIS_YYZ] = a[BASIS_YY] * b.z + a[BASIS_YZ] * b.y;
    res[BASIS_YZZ] = a[BASIS_YZ] * b.z + a[BASIS_ZZ] * b.y;
    res[BASIS_ZZZ] = a[BASIS_ZZ] * b.z;
    res[BASIS_XX] = a[BASIS_XX] * b.w + a[BASIS_X] * b.x;
    res[BASIS_XY] = a[BASIS_XY] * b.w + a[BASIS_X] * b.y + a[BASIS_Y] * b.x;
    res[BASIS_XZ] = a[BASIS_XZ] * b.w + a[BASIS_X] * b.z + a[BASIS_Z] * b.x;
    res[BASIS_YY] = a[BASIS_YY] * b.w + a[BASIS_Y] * b.y;
    res[BASIS_YZ] = a[BASIS_YZ] * b.w + a[BASIS_Y] * b.z + a[BASIS_Z] * b.y;
    res[BASIS_ZZ] = a[BASIS_ZZ] * b.w + a[BASIS_Z] * b.z;
    res[BASIS_X] = a[BASIS_X] * b.w + a[BASIS_1] * b.x;
    res[BASIS_Y] = a[BASIS_Y] * b.w + a[BASIS_1] * b.y;
    res[BASIS_Z] = a[BASIS_Z] * b.w + a[BASIS_1] * b.z;
    res[BASIS_1] = a[BASIS_1] * b.w;
    return res;
}

fn five_points_polynomial_constraints(nullspace: &NullspaceMat) -> ConstraintMat {
    // Build the polynomial form of E (equation (8) in Stewenius et al. [1])
    let mut e = [[Vector4::zeros(); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let x = nullspace[(3 * i + j, 0)];
            let y = nullspace[(3 * i + j, 1)];
            let z = nullspace[(3 * i + j, 2)];
            let w = nullspace[(3 * i + j, 3)];
            e[i][j] = Vector4::new(x, y, z, w);
        }
    }

    // The constraint matrix.
    let mut m = ConstraintMat::zeros();
    // Determinant constraint det(E) = 0; equation (19) of Nister [2].
    m.row_mut(0).copy_from(
        &(o2(o1(e[0][1], e[1][2]) - o1(e[0][2], e[1][1]), e[2][0])
            + o2(o1(e[0][2], e[1][0]) - o1(e[0][0], e[1][2]), e[2][1])
            + o2(o1(e[0][0], e[1][1]) - o1(e[0][1], e[1][0]), e[2][2]))
        .transpose(),
    );

    // Cubic singular values constraint.
    // Equation (20).
    let mut eet = [[PolyBasisVec::zeros(); 3]; 3];
    for i in 0..3 {
        // Since EET is symmetric, we only compute
        for j in 0..3 {
            // its upper triangular part.
            if i <= j {
                eet[i][j] = o1(e[i][0], e[j][0]) + o1(e[i][1], e[j][1]) + o1(e[i][2], e[j][2]);
            } else {
                eet[i][j] = eet[j][i];
            }
        }
    }

    // Equation (21).
    let mut l = eet;
    let trace = 0.5 * (eet[0][0] + eet[1][1] + eet[2][2]);
    for i in 0..3 {
        l[i][i] -= trace;
    }

    // Equation (23).
    for i in 0..3 {
        for j in 0..3 {
            let leij = o2(l[i][0], e[0][j]) + o2(l[i][1], e[1][j]) + o2(l[i][2], e[2][j]);
            m.row_mut(1 + i * 3 + j).copy_from(&leij.transpose());
        }
    }

    m
}

fn essentials_from_action_ebasis(
    at: &Square10,
    eb: &NullspaceMat,
) -> impl Iterator<Item = EssentialMatrix> {
    let eigenvalues = at.complex_eigenvalues();
    eigenvalues
        .iter()
        .filter_map(|e| {
            if e.im == 0.0 {
                let e = e.re;
                // Solve for the eigen vector.
                let li = Square10::from_diagonal_element(e);
                let lu = (at - li).full_piv_lu();
                lu.solve(&VectorN::<f32, U10>::zeros())
                    .map(|v| v.fixed_rows::<U4>(5).into_owned())
            } else {
                None
            }
        })
        .map(|vector| Matrix3::from_iterator((eb * vector).iter().copied()).transpose())
        .map(EssentialMatrix)
        .collect::<Vec<_>>()
        .into_iter()
}

pub fn five_points_relative_pose(x1: &Input, x2: &Input) -> impl Iterator<Item = EssentialMatrix> {
    // Step 1: Nullspace Extraction.
    let e_basis = if let Some(m) = five_points_nullspace_basis(x1, x2) {
        m
    } else {
        return essentials_from_action_ebasis(&Square10::zeros(), &NullspaceMat::zeros());
    };

    // Step 2: Constraint Expansion.
    let e_constraints = five_points_polynomial_constraints(&e_basis);

    // Step 3: Gauss-Jordan Elimination (done thanks to a LU decomposition).
    let c_lu = e_constraints.fixed_slice::<U10, U10>(0, 0).full_piv_lu();
    let m = if let Some(m) = c_lu.solve(&e_constraints.fixed_slice::<U10, U10>(0, 10).into_owned())
    {
        m
    } else {
        return essentials_from_action_ebasis(&Square10::zeros(), &NullspaceMat::zeros());
    };

    // For next steps we follow the matlab code given in Stewenius et al [1].

    // Build action matrix.

    let mut at = Square10::zeros();
    at.fixed_slice_mut::<U3, U10>(0, 0)
        .copy_from(&m.fixed_slice::<U3, U10>(0, 0));

    at.row_mut(3).copy_from(&m.row(4));
    at.row_mut(4).copy_from(&m.row(5));
    at.row_mut(5).copy_from(&m.row(7));
    at[(6, 0)] = -1.0;
    at[(7, 1)] = -1.0;
    at[(8, 3)] = -1.0;
    at[(9, 6)] = -1.0;

    essentials_from_action_ebasis(&at, &e_basis)
}
