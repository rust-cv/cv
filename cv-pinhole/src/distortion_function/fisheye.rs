use super::DistortionFunction;
use cv_core::nalgebra::{
    allocator::Allocator, storage::Storage, zero, ArrayStorage, DefaultAllocator, Dim, DimName,
    NamedDim, Vector, Vector1, VectorN, U1,
};
use num_traits::{Float, Zero};

/// Parametric fisheye radial distortion function.
///
/// Implement non-rectilinear (i.e. fisheye) projections. Perhaps using the Fisheye Factor from
///
/// $$
/// R = \begin{cases}
///   \frac {f}{k} ⋅ \tan \p{k ⋅ θ} & 0 < k ≤ 1 \\\\
///   f ⋅ θ  & k = 0 \\\\
///   \frac {f}{k} ⋅\sin \p{k ⋅ θ}  & -1 ≤ k < 0 \\\\
/// \end{cases}
/// $$
///
/// Varying $k ∈ [-1, 1]$ will gradually transform from rectilinear to orthographic projection.
///
/// substituting $θ = \arctan r$:
///
/// $$
/// \frac R r = \frac f r ⋅ \begin{cases}
///   \frac {1}{k} ⋅ \tan \p{k ⋅ \arctan r} & 0 < k ≤ 1 \\\\
///   \arctan r  & k = 0 \\\\
///   \frac {1}{k} ⋅\sin \p{k ⋅ \arctan r}  & -1 ≤ k < 0 \\\\
/// \end{cases}
/// $$
///
/// $$
/// \frac R r = \frac f r ⋅ \begin{cases}
///   r & k = 1 \\\\
///   \frac {1}{k} ⋅ \tan \p{k ⋅ \arctan r} & \frac 12 < k < 1 \\\\
///   \frac {1}{k} ⋅ \tan \p{k ⋅ \arctan r} & k = \frac 12 \\\\
///   \frac {1}{k} ⋅ \tan \p{k ⋅ \arctan r} & 0 < k < \frac 12 \\\\
///   \arctan r  & k = 0 \\\\
///   \frac {1}{k} ⋅\sin \p{k ⋅ \arctan r}  & -1 ≤ k < 0 \\\\
/// \end{cases}
/// $$
///
/// We can factor this as $R = f ⋅ f(θ, k)$ where the function $f$ is a distortion function:
///
/// $$
/// f(x, \vec β) = \begin{cases}
///   \frac {1}{β_0} ⋅ \tan \p{β_0 ⋅ x} & 0 < β_0 ≤ 1 \\\\
///   x  & β_0 = 0 \\\\
///   \frac {1}{β_0} ⋅\sin \p{β_0 ⋅ x}  & -1 ≤ β_0 < 0 \\\\
/// \end{cases}
/// $$
///
///
/// # References
///
/// * Wikipedia Fisheye Lens. [][wiki]
/// * Lensfun documentation.
/// * PTGui v11 Support question 3.28. [link][ptgui].
/// * Panotools wiki. [link][panotools].
///
/// [wiki]: https://en.wikipedia.org/wiki/Fisheye_lens
/// [opencv]: https://docs.opencv.org/master/db/d58/group__calib3d__fisheye.html
/// [lensfun]: https://lensfun.github.io/manual/latest/corrections.html
/// [ptgui]: https://www.ptgui.com/support.html#3_28
/// [panotools]: https://wiki.panotools.org/Fisheye_Projection
///   
///
#[derive(Clone, PartialEq, Debug)]
pub struct Fisheye(f64);

impl Default for Fisheye {
    fn default() -> Self {
        Fisheye(1.0)
    }
}

impl DistortionFunction for Fisheye {
    type NumParameters = U1;

    fn from_parameters<S>(parameters: Vector<f64, Self::NumParameters, S>) -> Self
    where
        S: Storage<f64, Self::NumParameters>,
    {
        Self(parameters[0])
    }

    fn parameters(&self) -> VectorN<f64, Self::NumParameters> {
        Vector1::new(self.0)
    }

    fn evaluate(&self, value: f64) -> f64 {
        todo!()
    }

    fn derivative(&self, value: f64) -> f64 {
        self.with_derivative(value).1
    }

    /// Simultaneously compute value and first derivative.
    fn with_derivative(&self, value: f64) -> (f64, f64) {
        todo!()
    }

    fn inverse(&self, value: f64) -> f64 {
        todo!()
    }

    fn gradient(&self, value: f64) -> VectorN<f64, Self::NumParameters> {
        todo!()
    }
}
