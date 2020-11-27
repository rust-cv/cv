use super::polynomial::Polynomial;
use super::DistortionFunction;
use cv_core::nalgebra::{
    allocator::Allocator, storage::Storage, DefaultAllocator, Dim, DimAdd, DimName, DimSum,
    SliceStorage, Vector, VectorN, U1,
};
use num_traits::Zero;

/// Rational distortion function.
///
/// $$
/// f(x, \vec β) = \frac{
///     β_0 + β_1 ​⋅ x + β_2 ​⋅ x^2 + ⋯ + β_n ⋅ x^n}{
///     β_{n+1} + β_{n+2} ​⋅ x + β_{n+3} ​⋅ x^2 + ⋯ + β_{n+1 +m} ⋅ x^{m}
/// }
/// $$
///
/// Where $n = \mathtt{DP} - 1$ and $m = \mathtt{DQ} - 1$.
///
/// See [`impl DistortionFunction`](#impl-DistortionFunction).
///
///
#[derive(Clone, PartialEq, Debug)]
pub struct Rational<DP: Dim, DQ: Dim>(Polynomial<DP>, Polynomial<DQ>)
where
    DP: DimAdd<DQ>,
    DefaultAllocator: Allocator<f64, DP>,
    DefaultAllocator: Allocator<f64, DQ>,
    DefaultAllocator: Allocator<f64, DimSum<DP, DQ>>;

impl<DP: Dim, DQ: Dim> Default for Rational<DP, DQ>
where
    DP: DimName,
    DQ: DimName,
    DP: DimAdd<DQ>,
    DimSum<DP, DQ>: DimName,
    DefaultAllocator: Allocator<f64, DP>,
    DefaultAllocator: Allocator<f64, DQ>,
    DefaultAllocator: Allocator<f64, DimSum<DP, DQ>>,
    Polynomial<DP>: Default,
    Polynomial<DQ>: Default,
{
    fn default() -> Self {
        Self(Polynomial::default(), Polynomial::default())
    }
}

impl<DP: Dim, DQ: Dim> DistortionFunction for Rational<DP, DQ>
where
    DP: DimName,
    DQ: DimName,
    DP: DimAdd<DQ>,
    DimSum<DP, DQ>: DimName,
    DefaultAllocator: Allocator<f64, DP>,
    DefaultAllocator: Allocator<f64, DQ>,
    DefaultAllocator: Allocator<f64, DimSum<DP, DQ>>,
{
    type NumParameters = DimSum<DP, DQ>;

    fn evaluate(&self, value: f64) -> f64 {
        let p = self.0.evaluate(value);
        let q = self.1.evaluate(value);
        p / q
    }

    fn derivative(&self, value: f64) -> f64 {
        self.with_derivative(value).1
    }

    /// Numerically invert the function.
    ///
    /// # Method
    ///
    /// Starting with the function definition:
    ///
    /// $$
    /// y = \frac{P(x)}{Q(x)}
    /// $$
    ///
    /// $$
    /// \frac{P'(x)}{Q(x)} - \frac{P(x)}{Q(x)^2} ⋅ Q'(x)
    /// $$
    ///
    /// $$
    /// \frac{P'(x) ⋅ Q(x) - P(x) ⋅ Q'(x)}{Q(x)^2}
    /// $$
    ///
    fn with_derivative(&self, value: f64) -> (f64, f64) {
        let (p, dp) = self.0.with_derivative(value);
        let (q, dq) = self.1.with_derivative(value);
        (p / q, (dp * q - p * dq) / (q * q))
    }

    fn parameters(&self) -> VectorN<f64, Self::NumParameters> {
        stack(self.0.parameters(), self.1.parameters())
    }

    fn from_parameters<S>(parameters: Vector<f64, Self::NumParameters, S>) -> Self
    where
        S: Storage<f64, Self::NumParameters>,
    {
        let (pp, pq) = unstack(&parameters);
        Self(
            Polynomial::from_parameters(pp),
            Polynomial::from_parameters(pq),
        )
    }

    /// Parameter gradient
    ///
    /// # Method
    ///
    /// $$
    /// ∇_{\vec β​} \frac{P(x, \vec β​_p)}{Q(x, \vec β​_q)}
    /// $$
    ///
    /// $$
    /// \p{∇_{\vec β​} P(x, \vec β​_p)} ⋅ \frac 1{Q(x, \vec β​_q)} -
    /// \p{∇_{\vec β​} Q(x, \vec β​_q)} ⋅ \frac{P(x, \vec β​_p)}{Q(x, \vec β​_q)^2}
    /// $$
    ///
    fn gradient(&self, value: f64) -> VectorN<f64, Self::NumParameters> {
        let p = self.0.evaluate(value);
        let q = self.1.evaluate(value);
        stack(
            self.0.gradient(value) * (1.0 / q),
            self.1.gradient(value) * (-p / (q * q)),
        )
    }
}

/// Stack two statically sized nalgebra vectors. Allocates a new vector for the
/// result on the stack.
fn stack<D1, D2, S1, S2>(
    vec1: Vector<f64, D1, S1>,
    vec2: Vector<f64, D2, S2>,
) -> VectorN<f64, DimSum<D1, D2>>
where
    D1: DimName,
    D2: DimName,
    S1: Storage<f64, D1>,
    S2: Storage<f64, D2>,
    D1: DimAdd<D2>,
    DimSum<D1, D2>: DimName,
    DefaultAllocator: Allocator<f64, DimSum<D1, D2>>,
{
    let mut result = VectorN::<f64, DimSum<D1, D2>>::zero();
    result.fixed_slice_mut::<D1, U1>(0, 0).copy_from(&vec1);
    result
        .fixed_slice_mut::<D2, U1>(D1::dim(), 0)
        .copy_from(&vec2);
    result
}

/// Split a statically sized nalgebra vector in two. Returns vectors that reference
/// slices of the input vector.
fn unstack<'a, D1, D2, S>(
    vector: &'a Vector<f64, DimSum<D1, D2>, S>,
) -> (
    Vector<f64, D1, SliceStorage<'a, f64, D1, U1, S::RStride, S::CStride>>,
    Vector<f64, D2, SliceStorage<'a, f64, D2, U1, S::RStride, S::CStride>>,
)
where
    D1: DimName,
    D2: DimName,
    S: Storage<f64, DimSum<D1, D2>>,
    D1: DimAdd<D2>,
    DimSum<D1, D2>: DimName,
{
    (
        vector.fixed_slice::<D1, U1>(0, 0),
        vector.fixed_slice::<D2, U1>(D1::dim(), 0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distortion_function::test::TestFloat;
    use crate::distortion_test_generate;
    use cv_core::nalgebra::{VectorN, U4, U8};
    use float_eq::assert_float_eq;
    use proptest::prelude::*;
    use rug::ops::Pow;
    use rug::Float;

    impl<DP: Dim, DQ: Dim> TestFloat for Rational<DP, DQ>
    where
        DP: DimName,
        DQ: DimName,
        DP: DimAdd<DQ>,
        DimSum<DP, DQ>: DimName,
        DefaultAllocator: Allocator<f64, DP>,
        DefaultAllocator: Allocator<f64, DQ>,
        DefaultAllocator: Allocator<f64, DimSum<DP, DQ>>,
        Polynomial<DP>: TestFloat<NumParameters = DP>,
        Polynomial<DQ>: TestFloat<NumParameters = DQ>,
    {
        fn evaluate_float(&self, x: &Float) -> Float {
            self.0.evaluate_float(x) / self.1.evaluate_float(x)
        }
    }

    /// OpenCV12 radial distortion for a GoPro Hero 6
    #[rustfmt::skip]
    fn functions(_index: usize) -> Rational<U4, U4> {
        Rational::from_parameters(
            VectorN::from_column_slice_generic(U8, U1, &[
                 1.0, 25.67400161236561, 15.249676433312018, 0.6729530830603175,
                 1.0, 25.95447969279203, 22.421345314744390, 3.0431830552169914,
            ])
        )
    }

    fn function() -> impl Strategy<Value = Rational<U4, U4>> {
        (0_usize..1).prop_map(functions)
    }

    distortion_test_generate!(
        function(),
        evaluate_eps = 3.0,
        derivative_eps = 25.0,
        inverse_eps = 2.0,
    );

    #[test]
    fn test_evaluate_literal() {
        let f = functions(0);
        let x = 0.06987809296337355;
        let value = f.evaluate(x);
        assert_float_eq!(value, 0.9810452524397972, ulps <= 0);
    }
}
