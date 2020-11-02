use super::DistortionFunction;
use cv_core::nalgebra::{DimAdd, DimSum};
use num_traits::Float;

#[derive(Clone, Debug, Default)]
struct Rational<P, Q>(P, Q)
where
    P: DistortionFunction,
    Q: DistortionFunction;

impl<P, Q> DistortionFunction for Rational<P, Q>
where
    P: DistortionFunction,
    Q: DistortionFunction,
    P::NumParameters: DimAdd<Q::NumParameters>,
{
    type NumParameters = DimSum<P::NumParameters, Q::NumParameters>;

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
}
