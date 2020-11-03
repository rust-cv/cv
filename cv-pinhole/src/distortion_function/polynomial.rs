// TODO: Currently the polynomial is expressed in coefficient form, but for
// numerical evaluation other forms such as Chebyshev or Lagrange may be more
// accurate.

use super::DistortionFunction;
use cv_core::nalgebra::{
    allocator::Allocator, storage::Storage, zero, ArrayStorage, DefaultAllocator, Dim, DimName,
    NamedDim, Vector, VectorN, U1,
};
use num_traits::{Float, Zero};

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
        for &coefficient in self.0.iter() {
            result *= value;
            result += coefficient;
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
        for &coefficient in self.0.iter() {
            derivative *= value;
            derivative += result;
            result *= value;
            result += coefficient;
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
        todo!()
    }
}
