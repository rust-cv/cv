use super::DistortionFunction;
use cv_core::nalgebra::{storage::Storage, Vector, Vector1, VectorN, U1};

/// Parametric fisheye radial distortion function.
///
/// Implements various projection functions using the generalized relation with a single
/// parameter $k$.
///
/// $$
/// r' = \\begin{cases}
///   \sin \p{k ⋅ \arctan r} ⋅ \frac 1 k & k < 0 \\\\
///   \arctan r & k = 0 \\\\
///   \tan \p{k ⋅ \arctan r}  ⋅ \frac 1 k & k > 0 \\\\
/// \end{cases}
/// $$
///
/// Varying $k ∈ [-1, 1]$ will gradually transform from orthographic to rectilinear projection. In
/// particular for $k = 1$ the equation simplifies to $r' = r$ representing the non-Fisheye
/// rectilinear projection. Other named values are:
///
/// | $k$         | name          | projection            |
/// |------------:|---------------|-----------------------|
/// | $-1$        | orthographic  | $r =  \sin θ$         |
/// | $- 1/2$ | equisolid     | $r = 2 \sin  θ/2$ |
/// | $0$         | equidistant   | $r = \tan θ$          |
/// | $1/2$  | stereographic | $r = 2 \tan  θ/2$ |
/// | $1$         | rectilinear   | $r = \tan θ$          |
///
/// Wikipedia has an excellent [comparison][wiki] of the different projections.
/// # References
///
/// * Wikipedia Fisheye Lens. [link][wiki].
/// * Lensfun documentation. [link][lensfun].
/// * PTGui v11 Support question 3.28. [link][ptgui].
/// * Panotools wiki. [link][panotools].
///
/// [wiki]: https://en.wikipedia.org/wiki/Fisheye_lens#Mapping_function
/// [opencv]: https://docs.opencv.org/master/db/d58/group__calib3d__fisheye.html
/// [lensfun]: https://lensfun.github.io/manual/latest/corrections.html
/// [ptgui]: https://www.ptgui.com/support.html#3_28
/// [panotools]: https://wiki.panotools.org/Fisheye_Projection
///   
///
#[derive(Clone, PartialEq, PartialOrd, Debug)]
pub struct Fisheye(f64);

impl Default for Fisheye {
    /// Defaults to rectilinear (non-Fisheye) projection.
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
        match self.0 {
            k if k < 0.0 => (k * value.atan()).sin() / k,
            k if k == 0.0 => value.atan(),
            k if k < 1.0 => (k * value.atan()).tan() / k,
            _ => value,
        }
    }

    fn derivative(&self, value: f64) -> f64 {
        match self.0 {
            k if k < 0.0 => (k * value.atan()).cos() / (1.0 + value.powi(2)),
            k if k == 0.0 => 1.0 / (1.0 + value.powi(2)),
            k if k < 1.0 => (k * value.atan()).cos().powi(-2) / (1.0 + value.powi(2)),
            _ => 1.0,
        }
    }

    fn inverse(&self, value: f64) -> f64 {
        match self.0 {
            k if k < 0.0 => ((k * value).asin() / k).tan(),
            k if k == 0.0 => value.tan(),
            k if k < 1.0 => ((k * value).atan() / k).tan(),
            _ => value,
        }
    }

    fn gradient(&self, _value: f64) -> VectorN<f64, Self::NumParameters> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distortion_function::test::TestFloat;
    use crate::distortion_test_generate;
    use proptest::prelude::*;
    use rug::Float;

    impl TestFloat for Fisheye {
        fn evaluate_float(&self, x: &Float) -> Float {
            let x = x.clone();
            match self.0 {
                k if k < 0.0 => (k * x.atan()).sin() / k,
                k if k == 0.0 => x.atan(),
                k if k < 1.0 => (k * x.atan()).tan() / k,
                _ => x,
            }
        }
    }

    fn function() -> impl Strategy<Value = Fisheye> {
        (-1_f64..1.).prop_map(Fisheye)
    }

    distortion_test_generate!(
        function(),
        evaluate_eps = 6.0,
        derivative_eps = 9.0,
        inverse_eps = 4.0,
    );
}
