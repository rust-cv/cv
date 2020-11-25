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
/// rectilinear projection.
///
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
    use cv_core::nalgebra::Vector1;
    use float_eq::assert_float_eq;
    use proptest::prelude::*;

    #[test]
    fn test_finite_difference() {
        let h = f64::EPSILON.powf(1.0 / 3.0);
        let test = |k, r| {
            let fisheye = Fisheye::from_parameters(Vector1::new(k));
            let h = f64::max(h * 0.1, h * r);
            let deriv = fisheye.derivative(r);
            let approx = (fisheye.evaluate(r + h) - fisheye.evaluate(r - h)) / (2.0 * h);
            assert_float_eq!(deriv, approx, rmax <= 2e2 * h * h);
        };
        proptest!(|(k in -1.0..1.0, r in 0.0..1.0)| {
            test(k, r);
        });
        proptest!(|(r in 0.0..1.0)| {
            test(0.0, r);
        });
        proptest!(|(r in 0.0..1.0)| {
            test(1.0, r);
        });
    }

    #[test]
    fn test_roundtrip_forward() {
        let test = |k, r| {
            let fisheye = Fisheye::from_parameters(Vector1::new(k));
            let eval = fisheye.evaluate(r);
            let inv = fisheye.inverse(eval);
            assert_float_eq!(inv, r, rmax <= 1e1 * f64::EPSILON);
        };
        proptest!(|(k in -1.0..1.0, r in 0.0..2.0)| {
            test(k, r);
        });
        proptest!(|(r in 0.0..1.0)| {
            test(0.0, r);
        });
        proptest!(|(r in 0.0..1.0)| {
            test(1.0, r);
        });
    }

    #[test]
    fn test_roundtrip_reverse() {
        let test = |k, r| {
            let fisheye = Fisheye::from_parameters(Vector1::new(k));
            let inv = fisheye.inverse(r);
            let eval = fisheye.evaluate(inv);
            assert_float_eq!(eval, r, rmax <= 1e1 * f64::EPSILON);
        };
        proptest!(|(k in -1.0..1.0, r in 0.0..0.75)| {
            test(k, r);
        });
        proptest!(|(k in -1.0..1.0, r in 0.0..0.75)| {
            test(0.0, r);
        });
        proptest!(|(k in -1.0..1.0, r in 0.0..0.75)| {
            test(1.0, r);
        });
    }

    // TODO: Test parameter gradient using finite differences.
}
