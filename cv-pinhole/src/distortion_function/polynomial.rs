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
/// * Make it work with `Degree = Dynamic`.
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
    use rug::Float;

    impl<Degree: Dim> Polynomial<Degree>
    where
        DefaultAllocator: Allocator<f64, Degree>,
    {
        /// Use RUG + MPFR to compute the nearest f64s to the exact values.
        pub fn with_derivative_exact(&self, x: f64) -> (f64, f64) {
            const PREC: u32 = 1000; // Compute using 1000 bit accuracy
            let x = Float::with_val(PREC, x);
            let mut value = Float::new(PREC);
            let mut derivative = Float::new(PREC);
            for i in (0..self.0.nrows()).rev() {
                derivative *= &x;
                derivative += &value;
                value *= &x;
                value += self.0[i];
            }
            (value.to_f64(), derivative.to_f64())
        }
    }

    #[rustfmt::skip]
    const POLYS: &[&[f64]] = &[
        &[1.0, 25.67400161236561, 15.249676433312018, 0.6729530830603175],
        &[1.0, 25.95447969279203, 22.421345314744390, 3.0431830552169914],
    ];

    fn polynomials(index: usize) -> Polynomial<U4> {
        Polynomial::from_parameters(Vector4::from_column_slice(POLYS[index]))
    }

    fn polynomial() -> impl Strategy<Value = Polynomial<U4>> {
        (0_usize..POLYS.len()).prop_map(polynomials)
    }

    #[test]
    fn test_evaluate_literal() {
        let f = polynomials(0);
        let x = 0.06987809296337355;
        let value = f.evaluate(x);
        assert_float_eq!(value, 2.86874326561548, ulps <= 0);
    }

    #[test]
    fn test_evaluate() {
        proptest!(|(f in polynomial(), x in 0.0..2.0)| {
            let value = f.evaluate(x);
            let expected = f.with_derivative_exact(x).0;
            assert_float_eq!(value, expected, rmax <= 2.0 * f64::EPSILON);
        });
    }

    #[test]
    fn test_with_derivative() {
        proptest!(|(f in polynomial(), x in 0.0..2.0)| {
            let value = f.with_derivative(x);
            let expected = f.with_derivative_exact(x);
            assert_float_eq!(value.0, expected.0, rmax <= 2.0 * f64::EPSILON);
            assert_float_eq!(value.1, expected.1, rmax <= 2.0 * f64::EPSILON);
        });
    }

    #[test]
    fn test_inverse() {
        proptest!(|(f in polynomial(), x in 0.0..2.0)| {
            let y = f.with_derivative_exact(x).0;
            let value = f.inverse(y);
            // There may be a multiple valid inverses, so instead of checking
            // the answer directly by `value == x`, we check that `f(value) == f(x)`.
            let y2 = f.with_derivative_exact(value).0;
            assert_float_eq!(y, y2, rmax <= 2.0 * f64::EPSILON);
        });
    }

    #[test]
    fn test_finite_difference() {
        let h = f64::EPSILON.powf(1.0 / 3.0);
        proptest!(|(f in polynomial(), x in 0.0..2.0)| {
            let h = f64::max(h * 0.1, h * x);
            let deriv = f.derivative(x);
            let approx = (f.evaluate(x + h) - f.evaluate(x - h)) / (2.0 * h);
            assert_float_eq!(deriv, approx, rmax <= 1e2 * h * h);
        });
    }

    #[test]
    fn test_roundtrip_forward() {
        proptest!(|(f in polynomial(), x in 0.0..2.0)| {
            let eval = f.evaluate(x);
            let inv = f.inverse(eval);
            assert_float_eq!(inv, x, rmax <= 1e4 * f64::EPSILON);
        });
    }

    #[test]
    fn test_roundtrip_reverse() {
        proptest!(|(f in polynomial(), x in 1.0..2.0)| {
            let inv = f.inverse(x);
            let eval = f.evaluate(inv);
            assert_float_eq!(eval, x, rmax <= 1e4 * f64::EPSILON);
        });
    }

    // TODO: Test parameter gradient using finite differences.
}
