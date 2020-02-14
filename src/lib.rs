#![no_std]

use cv_core::nalgebra::{self, Matrix3, MatrixMN, VectorN, U3, U8, U9};
use cv_core::sample_consensus::Estimator;
use cv_core::{EssentialMatrix, KeyPointsMatch};

fn encode_epipolar_equation(
    matches: impl Iterator<Item = KeyPointsMatch>,
) -> MatrixMN<f64, U8, U9> {
    let mut out: MatrixMN<f64, U8, U9> = nalgebra::zero();
    for (i, KeyPointsMatch(a, b)) in (0..8).zip(matches) {
        let mut row = VectorN::<f64, U9>::zeros();
        let ap = a.epipolar_point().0.coords;
        let bp = b.epipolar_point().0.coords;
        for j in 0..3 {
            let v = ap[j] * bp;
            row.fixed_rows_mut::<U3>(3 * j).copy_from(&v);
        }
        out.row_mut(i).copy_from(&row.transpose());
    }
    out
}

pub fn recondition_matrix(mat: Matrix3<f64>) -> EssentialMatrix {
    let old_svd = mat.svd(true, true);
    // We need to sort the singular values in the SVD.
    let mut sources = [0, 1, 2];
    sources.sort_unstable_by_key(|&ix| float_ord::FloatOrd(-old_svd.singular_values[ix]));
    let mut svd = old_svd;
    for (dest, &source) in sources.iter().enumerate() {
        svd.singular_values[dest] = old_svd.singular_values[source];
        svd.u
            .as_mut()
            .unwrap()
            .column_mut(dest)
            .copy_from(&old_svd.u.as_ref().unwrap().column(source));
        svd.v_t
            .as_mut()
            .unwrap()
            .row_mut(dest)
            .copy_from(&old_svd.v_t.as_ref().unwrap().row(source));
    }
    // Now that the singular values are sorted, find the closest
    // essential matrix to E in frobenius form.
    // This consists of averaging the two non-zero singular values
    // and zeroing out the near-zero singular value.
    svd.singular_values[2] = 0.0;
    let new_singular = (svd.singular_values[0] + svd.singular_values[1]) / 2.0;
    svd.singular_values[0] = new_singular;
    svd.singular_values[1] = new_singular;

    let mat = svd.recompose().unwrap();
    EssentialMatrix(mat)
}

pub struct EightPoint {
    pub epsilon: f64,
    pub iterations: usize,
}

impl EightPoint {
    pub fn new() -> Self {
        Default::default()
    }
}

impl Default for EightPoint {
    fn default() -> Self {
        Self {
            epsilon: 1e-9,
            iterations: 100,
        }
    }
}

impl Estimator<KeyPointsMatch> for EightPoint {
    type Model = EssentialMatrix;
    type ModelIter = Option<EssentialMatrix>;
    const MIN_SAMPLES: usize = 8;

    fn estimate<I>(&self, data: I) -> Self::ModelIter
    where
        I: Iterator<Item = KeyPointsMatch> + Clone,
    {
        let epipolar_constraint = encode_epipolar_equation(data);
        let eet = epipolar_constraint.transpose() * epipolar_constraint;
        let eigens = eet.try_symmetric_eigen(self.epsilon, self.iterations)?;
        let eigenvector = eigens
            .eigenvalues
            .iter()
            .enumerate()
            .min_by_key(|&(_, &n)| float_ord::FloatOrd(n))
            .map(|(ix, _)| eigens.eigenvectors.column(ix).into_owned())?;
        let mat = Matrix3::from_iterator(eigenvector.iter().copied());
        Some(EssentialMatrix(mat))
    }
}
