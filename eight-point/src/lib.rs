#![no_std]

use arrayvec::ArrayVec;
use cv_core::{
    nalgebra::{self, Matrix3, OMatrix, OVector, U8, U9},
    sample_consensus::Estimator,
    CameraToCamera, FeatureMatch,
};
use cv_pinhole::EssentialMatrix;

fn encode_epipolar_equation(matches: impl Iterator<Item = FeatureMatch>) -> OMatrix<f64, U8, U9> {
    let mut out: OMatrix<f64, U8, U9> = nalgebra::zero();
    for (i, FeatureMatch(a, b)) in (0..8).zip(matches) {
        let mut row = OVector::<f64, U9>::zeros();
        let ap = a.into_inner() / a.z;
        let bp = b.into_inner() / a.z;
        for j in 0..3 {
            let v = ap[j] * bp;
            row.fixed_rows_mut::<3>(3 * j).copy_from(&v);
        }
        out.row_mut(i).copy_from(&row.transpose());
    }
    out
}

/// Performs the
/// [eight-point algorithm](https://en.wikipedia.org/wiki/Eight-point_algorithm)
/// by Richard Hartley and Andrew Zisserman.
///
/// To recondition the matrix produced by estimation, see
/// [`cv_pinhole::EssentialMatrix::recondition`].
#[derive(Copy, Clone, Debug)]
pub struct EightPoint {
    pub epsilon: f64,
    pub iterations: usize,
}

impl EightPoint {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn from_matches<I>(&self, data: I) -> Option<EssentialMatrix>
    where
        I: Iterator<Item = FeatureMatch> + Clone,
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

impl Default for EightPoint {
    fn default() -> Self {
        Self {
            epsilon: 1e-12,
            iterations: 1000,
        }
    }
}

impl Estimator<FeatureMatch> for EightPoint {
    type Model = CameraToCamera;
    type ModelIter = ArrayVec<CameraToCamera, 4>;
    const MIN_SAMPLES: usize = 8;

    fn estimate<I>(&self, data: I) -> Self::ModelIter
    where
        I: Iterator<Item = FeatureMatch> + Clone,
    {
        self.from_matches(data)
            .and_then(|essential| essential.possible_unscaled_poses(self.epsilon, self.iterations))
            .map(Into::into)
            .unwrap_or_default()
    }
}
