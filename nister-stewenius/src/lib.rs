#![no_std]

use arrayvec::ArrayVec;
use cv_core::{
    nalgebra::{
        self,
        dimension::{U10, U20, U4, U5, U9},
        DimName, Matrix3, OMatrix, OVector, UnitVector3, Vector3, Vector4,
    },
    CameraToCamera, FeatureMatch,
};
use cv_pinhole::EssentialMatrix;
use sample_consensus::Estimator;

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

const EIGEN_CONVERGENCE: f64 = 1e-12;
const EIGEN_ITERATIONS: usize = 1000;
const EIGEN_THRESHOLD: f64 = 1e-12;
const SVD_CONVERGENCE: f64 = 1e-12;
const SVD_ITERATIONS: usize = 1000;
/// The threshold which the singular value must be below for it
/// to be considered the null-space.
const SVD_NULL_THRESHOLD: f64 = 1e-12;

type PolyBasisVec = OVector<f64, U20>;
type NullspaceMat = OMatrix<f64, U9, U4>;
type ConstraintMat = OMatrix<f64, U10, U20>;
type Square10 = OMatrix<f64, U10, U10>;

fn encode_epipolar_equation(
    a: &[UnitVector3<f64>; 5],
    b: &[UnitVector3<f64>; 5],
) -> OMatrix<f64, U5, U9> {
    let mut out: OMatrix<f64, U5, U9> = nalgebra::zero();
    for i in 0..U5::dim() {
        let mut row = OVector::<f64, U9>::zeros();
        let ap = a[i].into_inner();
        let bp = b[i].into_inner();
        for j in 0..3 {
            let v = ap[j] * bp;
            row.fixed_rows_mut::<3>(3 * j).copy_from(&v);
        }
        out.row_mut(i).copy_from(&row.transpose());
    }
    out
}

pub fn five_points_nullspace_basis(
    a: &[UnitVector3<f64>; 5],
    b: &[UnitVector3<f64>; 5],
) -> Option<NullspaceMat> {
    let epipolar_constraint = encode_epipolar_equation(a, b);
    let ee = epipolar_constraint.transpose() * epipolar_constraint;
    ee.try_symmetric_eigen(EIGEN_CONVERGENCE, EIGEN_ITERATIONS)
        .and_then(|m| {
            // We need to sort the eigenvectors by their corresponding eigenvalue.
            let mut sources = [0, 1, 2, 3, 4, 5, 6, 7, 8];
            sources.sort_unstable_by_key(|&ix| float_ord::FloatOrd(m.eigenvalues[ix]));
            let mut nullspace = NullspaceMat::zeros();
            // We must have a nullity of 4.
            let nullity = sources
                .iter()
                .map(|&i| m.eigenvalues[i])
                .enumerate()
                .find(|&(_, e)| e > EIGEN_THRESHOLD)
                .map(|(ix, _)| ix)?;
            if nullity != 4 {
                return None;
            }
            // Place the null space vectors into the null matrix.
            for (&ix, mut column) in sources.iter().zip(nullspace.column_iter_mut()) {
                column.copy_from(&m.eigenvectors.column(ix));
            }
            Some(nullspace)
        })
}

fn o1(a: Vector4<f64>, b: Vector4<f64>) -> PolyBasisVec {
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

fn o2(a: PolyBasisVec, b: Vector4<f64>) -> PolyBasisVec {
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
    res
}

fn five_points_polynomial_constraints(nullspace: &NullspaceMat) -> ConstraintMat {
    // Build the polynomial form of E (equation (8) in Stewenius et al. [1])
    let mut e_poly = [[Vector4::zeros(); 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let x = nullspace[(3 * i + j, 0)];
            let y = nullspace[(3 * i + j, 1)];
            let z = nullspace[(3 * i + j, 2)];
            let w = nullspace[(3 * i + j, 3)];
            e_poly[i][j] = Vector4::new(x, y, z, w);
        }
    }

    // The constraint matrix.
    let mut m = ConstraintMat::zeros();
    // Determinant constraint det(E) = 0; equation (19) of Nister [2].
    m.row_mut(0).copy_from(
        &(o2(
            o1(e_poly[0][1], e_poly[1][2]) - o1(e_poly[0][2], e_poly[1][1]),
            e_poly[2][0],
        ) + o2(
            o1(e_poly[0][2], e_poly[1][0]) - o1(e_poly[0][0], e_poly[1][2]),
            e_poly[2][1],
        ) + o2(
            o1(e_poly[0][0], e_poly[1][1]) - o1(e_poly[0][1], e_poly[1][0]),
            e_poly[2][2],
        ))
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
                eet[i][j] = o1(e_poly[i][0], e_poly[j][0])
                    + o1(e_poly[i][1], e_poly[j][1])
                    + o1(e_poly[i][2], e_poly[j][2]);
            } else {
                eet[i][j] = eet[j][i];
            }
        }
    }

    // Equation (21).
    let mut l = eet;
    let trace = 0.5 * (eet[0][0] + eet[1][1] + eet[2][2]);
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        l[i][i] -= trace;
    }

    // Equation (23).
    #[allow(clippy::needless_range_loop)]
    for i in 0..3 {
        for j in 0..3 {
            let leij =
                o2(l[i][0], e_poly[0][j]) + o2(l[i][1], e_poly[1][j]) + o2(l[i][2], e_poly[2][j]);
            m.row_mut(1 + i * 3 + j).copy_from(&leij.transpose());
        }
    }

    m
}

fn compute_eigenvector(m: &Square10, lambda: f64) -> Option<OVector<f64, U10>> {
    (m - Square10::from_diagonal_element(lambda))
        .try_svd(false, true, SVD_CONVERGENCE, SVD_ITERATIONS)
        .and_then(|svd| {
            // Ensure that the singular value is below a threshold.
            if svd.singular_values[9] < SVD_NULL_THRESHOLD {
                Some(svd.v_t?.row(9).transpose())
            } else {
                None
            }
        })
}

fn essentials_from_action_ebasis(
    at: Square10,
    eb: NullspaceMat,
) -> impl Iterator<Item = EssentialMatrix> {
    let eigenvalues = at.complex_eigenvalues();
    (0..eigenvalues.len())
        .filter_map(move |i| {
            let e = eigenvalues[i];
            if e.im == 0.0 {
                let e = e.re;
                // Solve for the eigen vector.
                compute_eigenvector(&at, e).map(|v| v.fixed_rows::<4>(5).into_owned())
            } else {
                None
            }
        })
        .map(move |vector| Matrix3::from_iterator((eb * vector).iter().copied()))
        .map(EssentialMatrix)
}

/// Takes in two sets of normalized key points.
/// Returns all essential matrix solutions.
fn five_points_relative_pose(
    a: &[UnitVector3<f64>; 5],
    b: &[UnitVector3<f64>; 5],
) -> impl Iterator<Item = EssentialMatrix> {
    // Step 1: Nullspace Extraction.
    let e_basis = if let Some(m) = five_points_nullspace_basis(a, b) {
        m
    } else {
        return essentials_from_action_ebasis(Square10::zeros(), NullspaceMat::zeros());
    };

    // Step 2: Constraint Expansion.
    let e_constraints = five_points_polynomial_constraints(&e_basis);

    // Step 3: Gauss-Jordan Elimination (done thanks to a LU decomposition).
    let c_lu = e_constraints.fixed_slice::<10, 10>(0, 0).full_piv_lu();
    let m = if let Some(m) = c_lu.solve(&e_constraints.fixed_slice::<10, 10>(0, 10).into_owned()) {
        m
    } else {
        return essentials_from_action_ebasis(Square10::zeros(), NullspaceMat::zeros());
    };

    // For next steps we follow the matlab code given in Stewenius et al [1].

    // Build action matrix.

    let mut at = Square10::zeros();
    at.fixed_slice_mut::<3, 10>(0, 0)
        .copy_from(&m.fixed_slice::<3, 10>(0, 0));

    at.row_mut(3).copy_from(&m.row(4));
    at.row_mut(4).copy_from(&m.row(5));
    at.row_mut(5).copy_from(&m.row(7));
    at[(6, 0)] = -1.0;
    at[(7, 1)] = -1.0;
    at[(8, 3)] = -1.0;
    at[(9, 6)] = -1.0;

    essentials_from_action_ebasis(at, e_basis)
}

/// Implements the 5-point algorithm from the paper "Recent developments on direct relative orientation".
pub struct NisterStewenius {
    pub epsilon: f64,
    pub iterations: usize,
}

impl NisterStewenius {
    pub fn new() -> Self {
        Self {
            epsilon: 1e-12,
            iterations: 1000,
        }
    }
}

impl Default for NisterStewenius {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator<FeatureMatch> for NisterStewenius {
    type Model = CameraToCamera;

    type ModelIter = ArrayVec<CameraToCamera, 40>;

    const MIN_SAMPLES: usize = 5;

    fn estimate<I>(&self, data: I) -> Self::ModelIter
    where
        I: Iterator<Item = FeatureMatch> + Clone,
    {
        let mut a = [UnitVector3::new_unchecked(Vector3::y()); 5];
        let mut b = [UnitVector3::new_unchecked(Vector3::y()); 5];
        let mut count = 0;
        for ((a, b), m) in a.iter_mut().zip(b.iter_mut()).zip(data) {
            *a = m.0;
            *b = m.1;
            count += 1;
        }
        assert!(count == 5);
        five_points_relative_pose(&a, &b)
            .filter_map(|essential| {
                essential.possible_unscaled_poses(self.epsilon, self.iterations)
            })
            .flat_map(IntoIterator::into_iter)
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn vec_to_poly_basis(v: Vector4<f64>) -> PolyBasisVec {
        let mut res = PolyBasisVec::zeros();
        res[BASIS_X] = v.x;
        res[BASIS_Y] = v.y;
        res[BASIS_Z] = v.z;
        res[BASIS_1] = v.w;
        res
    }

    fn eval_polynomial(p: PolyBasisVec, x: f64, y: f64, z: f64) -> f64 {
        p[BASIS_XXX] * x * x * x
            + p[BASIS_XXY] * x * x * y
            + p[BASIS_XXZ] * x * x * z
            + p[BASIS_XYY] * x * y * y
            + p[BASIS_XYZ] * x * y * z
            + p[BASIS_XZZ] * x * z * z
            + p[BASIS_YYY] * y * y * y
            + p[BASIS_YYZ] * y * y * z
            + p[BASIS_YZZ] * y * z * z
            + p[BASIS_ZZZ] * z * z * z
            + p[BASIS_XX] * x * x
            + p[BASIS_XY] * x * y
            + p[BASIS_XZ] * x * z
            + p[BASIS_YY] * y * y
            + p[BASIS_YZ] * y * z
            + p[BASIS_ZZ] * z * z
            + p[BASIS_X] * x
            + p[BASIS_Y] * y
            + p[BASIS_Z] * z
            + p[BASIS_1]
    }

    #[test]
    fn o1_manual() {
        let p1 = Vector4::new(0.1, 0.8, 0.3, 0.2);
        let p2 = Vector4::new(0.5, 0.45, 0.82, 0.15);
        let p3 = o1(p1, p2);
        for z in -5..5 {
            for y in -5..5 {
                for x in -5..5 {
                    let x = x as f64;
                    let y = y as f64;
                    let z = z as f64;
                    let o1_comp = eval_polynomial(p3, x, y, z);
                    let man_comp = eval_polynomial(vec_to_poly_basis(p1), x, y, z)
                        * eval_polynomial(vec_to_poly_basis(p2), x, y, z);
                    assert!((o1_comp - man_comp).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn o2_manual() {
        let mut p1 = PolyBasisVec::zeros();
        p1[BASIS_XX] = 0.2;
        p1[BASIS_XY] = 0.81;
        p1[BASIS_XZ] = 0.66;
        p1[BASIS_YY] = 0.91;
        p1[BASIS_YZ] = 0.88;
        p1[BASIS_ZZ] = 0.14;
        p1[BASIS_X] = 0.97;
        p1[BASIS_Y] = 0.3;
        p1[BASIS_Z] = 0.38;
        p1[BASIS_1] = 0.72;

        let p2 = Vector4::new(0.5, 0.45, 0.82, 0.15);
        let p3 = o2(p1, p2);
        for z in -5..5 {
            for y in -5..5 {
                for x in -5..5 {
                    let x = x as f64;
                    let y = y as f64;
                    let z = z as f64;
                    let o1_comp = eval_polynomial(p3, x, y, z);
                    let man_comp = eval_polynomial(p1, x, y, z)
                        * eval_polynomial(vec_to_poly_basis(p2), x, y, z);
                    assert!((o1_comp - man_comp).abs() < 1e-8);
                }
            }
        }
    }

    fn old_compute_eigenvector(m: &Square10, lambda: f64) -> Option<OVector<f64, U10>> {
        (m - Square10::from_diagonal_element(lambda))
            .try_svd_unordered(false, true, SVD_CONVERGENCE, SVD_ITERATIONS)
            .and_then(|svd| {
                // Find the lowest singular value's index and value.
                let (ix, &v) = svd
                    .singular_values
                    .iter()
                    .enumerate()
                    .min_by_key(|&(_, &v)| float_ord::FloatOrd(v))
                    .unwrap();
                // Ensure that the singular value is below a threshold.
                if v < SVD_NULL_THRESHOLD {
                    Some(svd.v_t.unwrap().row(ix).transpose().into_owned())
                } else {
                    None
                }
            })
    }

    #[test]
    fn test_compute_eigenvector() {
        for i in 0..100 {
            let s10 = Square10::new_random();
            // simple prng for testing purposes
            let lambda = (i as f64 * 12.3456789).sin();

            let old = old_compute_eigenvector(&s10, lambda);
            let new = compute_eigenvector(&s10, lambda);

            assert_eq!(old, new);
        }
    }
}
