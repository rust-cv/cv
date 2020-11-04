use super::polynomial::Polynomial;
use super::DistortionFunction;
use cv_core::nalgebra::{
    allocator::Allocator, storage::Storage, DefaultAllocator, Dim, DimAdd, DimName, DimSum,
    SliceStorage, Vector, VectorN, U1,
};
use num_traits::{Float, Zero};

#[derive(Clone, PartialEq, Debug)]
struct Rational<DP: Dim, DQ: Dim>(Polynomial<DP>, Polynomial<DQ>)
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
    fn inverse(&self, value: f64) -> f64 {
        // Maxmimum number of iterations to use in Newton-Raphson inversion.
        const MAX_ITERATIONS: usize = 100;

        // Convergence treshold for Newton-Raphson inversion.
        const EPSILON: f64 = f64::EPSILON;

        let mut x = value;
        for _ in 0..MAX_ITERATIONS {
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
