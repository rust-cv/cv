mod fisheye;
mod identity;
mod polynomial;
mod rational;

use cv_core::nalgebra::{
    allocator::Allocator, storage::Storage, DefaultAllocator, Dim, Vector, VectorN,
};

// Re-exports
#[doc(inline)]
pub use identity::Identity;

#[doc(inline)]
pub use polynomial::Polynomial;

#[doc(inline)]
pub use rational::Rational;

#[doc(inline)]
pub use fisheye::Fisheye;

/// Trait for parameterized functions specifying 1D distortions.
///
/// $$
/// y = f(x, \vec β)
/// $$
///
/// Provides evaluations, inverse, derivative and derivative with respect to
/// parameters.
///
/// The function $f$ is assumed to be monotonic.
///
/// # To do
///
/// * Generalize to arbitrary input/output dimensions.
///
pub trait DistortionFunction: Clone
where
    DefaultAllocator: Allocator<f64, Self::NumParameters>,
{
    /// The number of parameters, $\dim \vec β$ as a nalgebra type level integer.
    ///
    /// # To do
    ///
    /// * Make this [`DimName`](cv_core::nalgebra::DimName) or provide a method
    ///   to retrieve the dynamic value.
    type NumParameters: Dim;

    /// Create a new instance from parameters $\vec β$.
    fn from_parameters<S>(parameters: Vector<f64, Self::NumParameters, S>) -> Self
    where
        S: Storage<f64, Self::NumParameters>;

    /// Get function parameters $\vec β$.
    fn parameters(&self) -> VectorN<f64, Self::NumParameters>;

    /// Evaluate $f(\mathtt{value}, \vec β)$.
    fn evaluate(&self, value: f64) -> f64;

    /// Evaluate the derivative $f'(\mathtt{value}, \vec β)$ where $f' = \frac{\d}{\d x} f$.
    fn derivative(&self, value: f64) -> f64;

    /// Simultaneously evaluate function and its derivative.
    ///
    /// The default implementation combines the results from
    /// [`Self::evaluate`] and [`Self::derivative`]. When it is more efficient
    /// to evaluate them together this function can be implemented.
    fn with_derivative(&self, value: f64) -> (f64, f64) {
        (self.evaluate(value), self.derivative(value))
    }

    /// Evaluate the inverse $f^{-1}(\mathtt{value}, \vec β)$.
    fn inverse(&self, value: f64) -> f64;

    /// Parameter gradient $∇_{\vec β​} f(\mathtt{value}, \vec β)$.
    fn gradient(&self, value: f64) -> VectorN<f64, Self::NumParameters>;
}
