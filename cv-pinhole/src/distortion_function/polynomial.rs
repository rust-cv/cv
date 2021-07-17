// TODO: Currently the polynomial is expressed in coefficient form, but for
// numerical evaluation other forms such as Chebyshev or Lagrange may be more
// accurate.

use super::DistortionFunction;
use cv_core::nalgebra::{
    allocator::Allocator, storage::Storage, DefaultAllocator, Dim, Vector, VectorN, U1,
};
use num_traits::Zero;

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
    /// Uses Horner's method with it's algorithmic derivative.
    ///
    /// $$
    /// \begin{aligned}
    /// y_{i+1} &= y_i ⋅ x + c_{n - i} \\\\
    /// y'_{i+1} &= y'_i ⋅ x + y_i
    /// \end{aligned}
    /// $$
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
    use crate::distortion_function::test::TestFloat;
    use crate::distortion_test_generate;
    use cv_core::nalgebra::{DimName, Vector4, U4};
    use float_eq::assert_float_eq;
    use proptest::prelude::*;
    use rug::Float;

    impl<Degree> TestFloat for Polynomial<Degree>
    where
        Degree: Dim + DimName,
        DefaultAllocator: Allocator<f64, Degree>,
    {
        fn evaluate_float(&self, x: &Float) -> Float {
            let mut value = Float::new(x.prec());
            for i in (0..self.0.nrows()).rev() {
                value *= x;
                value += self.0[i];
            }
            value
        }
    }

    #[rustfmt::skip]
    const POLYS: &[&[f64]] = &[
        &[1.0, 25.67400161236561, 15.249676433312018, 0.6729530830603175],
        &[1.0, 25.95447969279203, 22.421345314744390, 3.0431830552169914],
        // &[0.0, -3.23278793e-03, 9.53176056e-05, 0.0],
        // &[0.0, -9.35687185e-05, 2.96341863e-05, 0.0],
    ];

    fn functions(index: usize) -> Polynomial<U4> {
        Polynomial::from_parameters(Vector4::from_column_slice(POLYS[index]))
    }

    fn function() -> impl Strategy<Value = Polynomial<U4>> {
        (0_usize..POLYS.len()).prop_map(functions)
    }

    distortion_test_generate!(
        function(),
        evaluate_eps = 2.0,
        derivative_eps = 3.0,
        inverse_eps = 2.0,
        gradient_eps = 1.0,
    );

    #[test]
    fn test_evaluate_literal() {
        let f = functions(0);
        let x = 0.06987809296337355;
        let value = f.evaluate(x);
        assert_float_eq!(value, 2.86874326561548, ulps <= 0);
    }

    // TODO: Test parameter gradient using finite differences.
}
