use super::polynomial::Polynomial;
use super::DistortionFunction;
use cv_core::nalgebra::{
    allocator::Allocator, storage::Storage, DefaultAllocator, Dim, DimAdd, DimName, DimSum,
    SliceStorage, Vector, VectorN, U1,
};
use num_traits::{Float, Zero};

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
///
#[derive(Clone, PartialEq, Debug)]
pub struct Rational<DP: Dim, DQ: Dim>(Polynomial<DP>, Polynomial<DQ>)
where
    DP: DimAdd<DQ>,
    DefaultAllocator: Allocator<f64, DP>,
    DefaultAllocator: Allocator<f64, DQ>,
    DefaultAllocator: Allocator<f64, DimSum<DP, DQ>>;

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

    fn with_derivative(&self, value: f64) -> (f64, f64) {
        let (p, dp) = self.0.with_derivative(value);
        let (q, dq) = self.1.with_derivative(value);
        (p / q, (dp * q - p * dq) / (q * q))
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
    /// We manipulate this into a root finding problem linear in $P$, $Q$ which
    /// gives $y ⋅ Q(x) - P(x) = 0$. To solve this we use the Newton-Raphson
    /// method with $x_0 = y$ and
    ///
    /// $$
    /// x_{i+1} = x_i - \frac{y ⋅ Q(x) - P(x)}{y ⋅ Q'(x) - P'(x)}
    /// $$
    ///
    /// # To do
    ///
    /// * Improve numerical precision. Currently the Newton-Raphson can oscillate
    ///   or diverge.
    ///
    fn inverse(&self, value: f64) -> f64 {
        // Maxmimum number of iterations to use in Newton-Raphson inversion.
        const MAX_ITERATIONS: usize = 20;

        // Convergence threshold for Newton-Raphson inversion.
        const EPSILON: f64 = f64::EPSILON;

        let mut x = value;
        for _i in 0..MAX_ITERATIONS {
            let (p, dp) = self.0.with_derivative(x);
            let (q, dq) = self.1.with_derivative(x);
            let delta = (value * q - p) / (value * dq - dp);
            x -= delta;
            if Float::abs(delta) <= EPSILON * x {
                break;
            }
        }
        x
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
    use cv_core::nalgebra::{VectorN, U4, U8};
    use float_eq::assert_float_eq;
    use proptest::prelude::*;

    /// OpenCV12 radial distortion for a GoPro Hero 6
    #[rustfmt::skip]
    fn rational_1() -> Rational<U4, U4> {
        Rational::from_parameters(
            VectorN::from_column_slice_generic(U8, U1, &[
                0.6729530830603175, 15.249676433312018, 25.67400161236561, 1.0,
                3.0431830552169914, 22.421345314744390, 25.95447969279203, 1.0,
            ])
        )
    }

    #[test]
    fn test_evaluate_1() {
        let radial = rational_1();
        let result = radial.evaluate(0.06987809296337355);
        assert_float_eq!(result, 0.9810452524397972, ulps <= 0);
    }

    #[test]
    fn test_inverse_1() {
        let radial = rational_1();
        let result = radial.inverse(0.9810452524397972);
        assert_float_eq!(result, 0.06987809296337355, ulps <= 8);
    }

    #[test]
    fn test_finite_difference_1() {
        let h = f64::EPSILON.powf(1.0 / 3.0);
        let radial = rational_1();
        proptest!(|(r in 0.0..2.0)| {
            let h = f64::max(h * 0.1, h * r);
            let deriv = radial.derivative(r);
            let approx = (radial.evaluate(r + h) - radial.evaluate(r - h)) / (2.0 * h);
            assert_float_eq!(deriv, approx, rmax <= 1e4 * h * h);
        });
    }

    #[test]
    fn test_roundtrip_forward_1() {
        let radial = rational_1();
        proptest!(|(r in 0.0..2.0)| {
            let eval = radial.evaluate(r);
            dbg!(r, eval);
            let inv = radial.inverse(eval);
            assert_float_eq!(inv, r, abs <= 1e8 * f64::EPSILON);
        });
    }

    #[test]
    fn test_roundtrip_reverse_1() {
        let radial = rational_1();
        proptest!(|(r in 0.0..2.0)| {
            let inv = radial.inverse(r);
            let eval = radial.evaluate(inv);
            assert_float_eq!(eval, r, ulps <= 150);
        });
    }
}
