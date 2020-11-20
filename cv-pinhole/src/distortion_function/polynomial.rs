// TODO: Currently the polynomial is expressed in coefficient form, but for
// numerical evaluation other forms such as Chebyshev or Lagrange may be more
// accurate.

use super::DistortionFunction;
use cv_core::nalgebra::{
    allocator::Allocator, storage::Storage, DefaultAllocator, Dim, Vector, VectorN, U1,
};
use num_traits::{Float, Zero};

/// Polynomial distortion function
///
/// $$
/// f(x, \vec β) = β_0 + β_1 ​⋅ x + β_2 ​⋅ x^2 + ⋯ + β_n ⋅ x^n
/// $$
///
/// # To do
///
/// * (Bug) Parameters are currently in reverse order.
///
#[derive(Clone, PartialEq, Debug)]
pub struct Polynomial<Degree: Dim>(VectorN<f64, Degree>)
where
    DefaultAllocator: Allocator<f64, Degree>;

impl<Degree: Dim> Default for Polynomial<Degree>
where
    DefaultAllocator: Allocator<f64, Degree>,
    VectorN<f64, Degree>: Zero,
{
    fn default() -> Self {
        Self(VectorN::zero())
    }
}

impl<Degree: Dim> DistortionFunction for Polynomial<Degree>
where
    DefaultAllocator: Allocator<f64, Degree>,
{
    type NumParameters = Degree;

    fn from_parameters<S>(parameters: Vector<f64, Self::NumParameters, S>) -> Self
    where
        S: Storage<f64, Self::NumParameters>,
    {
        Self(parameters.into_owned())
    }

    fn parameters(&self) -> VectorN<f64, Self::NumParameters> {
        self.0.clone()
    }

    fn evaluate(&self, value: f64) -> f64 {
        // Basic horner evaluation.
        // Ideally the compiler unrolls the loop and realizes that the first.
        // multiplication is redundant.
        let mut result = 0.0;
        for i in (0..self.0.nrows()).rev() {
            result *= value;
            result += self.0[i];
        }
        result
    }

    fn derivative(&self, value: f64) -> f64 {
        self.with_derivative(value).1
    }

    /// Simultaneously compute value and first derivative.
    ///
    /// # Method
    ///
    fn with_derivative(&self, value: f64) -> (f64, f64) {
        let mut result = 0.0;
        let mut derivative = 0.0;
        for i in (0..self.0.nrows()).rev() {
            derivative *= value;
            derivative += result;
            result *= value;
            result += self.0[i];
        }
        (result, derivative)
    }

    /// Numerically invert the function.
    ///
    /// # Method
    ///
    /// We solve for a root of $P(x) - y$ using the Newton-Raphson method with
    /// $x_0 = y$ and
    ///
    /// $$
    /// x_{i+1} = x_i - \frac{P(x) - value}{P'(x)}
    /// $$
    ///
    fn inverse(&self, value: f64) -> f64 {
        // Maxmimum number of iterations to use in Newton-Raphson inversion.
        const MAX_ITERATIONS: usize = 100;

        // Convergence treshold for Newton-Raphson inversion.
        const EPSILON: f64 = f64::EPSILON;

        let mut x = value;
        for _ in 0..MAX_ITERATIONS {
            let (p, dp) = self.with_derivative(x);
            let delta = (p - value) / dp;
            x -= delta;
            if Float::abs(delta) <= EPSILON * x {
                break;
            }
        }
        x
    }

    fn gradient(&self, value: f64) -> VectorN<f64, Self::NumParameters> {
        let mut factor = 1.0;
        VectorN::from_fn_generic(self.0.data.shape().0, U1, move |_, _| {
            let coefficient = factor;
            factor *= value;
            coefficient
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::nalgebra::{Vector4, U4};
    use float_eq::assert_float_eq;
    use proptest::prelude::*;

    fn polynomial_1() -> Polynomial<U4> {
        Polynomial::from_parameters(Vector4::new(
            1.0,
            25.67400161236561,
            15.249676433312018,
            0.6729530830603175,
        ))
    }

    #[test]
    fn test_evaluate_1() {
        let radial = polynomial_1();
        let result = radial.evaluate(0.06987809296337355);
        assert_float_eq!(result, 2.86874326561548, ulps <= 0);
    }

    #[test]
    fn test_inverse_1() {
        let radial = polynomial_1();
        let result = radial.inverse(2.86874326561548);
        assert_float_eq!(result, 0.06987809296337355, ulps <= 0);
    }

    #[test]
    fn test_finite_difference_1() {
        let h = f64::EPSILON.powf(1.0 / 3.0);
        let radial = polynomial_1();
        proptest!(|(r in 0.0..2.0)| {
            let h = f64::max(h * 0.1, h * r);
            let deriv = radial.derivative(r);
            let approx = (radial.evaluate(r + h) - radial.evaluate(r - h)) / (2.0 * h);
            assert_float_eq!(deriv, approx, rmax <= 1e2 * h * h);
        });
    }

    #[test]
    fn test_roundtrip_forward_1() {
        let radial = polynomial_1();
        proptest!(|(r in 0.0..2.0)| {
            let eval = radial.evaluate(r);
            let inv = radial.inverse(eval);
            assert_float_eq!(inv, r, rmax <= 1e4 * f64::EPSILON);
        });
    }

    #[test]
    fn test_roundtrip_reverse_1() {
        let radial = polynomial_1();
        proptest!(|(r in 0.0..2.0)| {
            let inv = radial.inverse(r);
            let eval = radial.evaluate(inv);
            assert_float_eq!(eval, r, rmax <= 1e4 * f64::EPSILON);
        });
    }
}
