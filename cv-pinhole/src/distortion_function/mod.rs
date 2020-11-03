mod identity;
mod polynomial;
mod rational;

use cv_core::nalgebra::{
    allocator::Allocator, storage::Storage, DefaultAllocator, Dim, Vector, VectorN,
};

pub trait DistortionFunction: Clone
where
    DefaultAllocator: Allocator<f64, Self::NumParameters>,
{
    /// Type level number of parameters
    type NumParameters: Dim;

    fn from_parameters<S>(parameters: Vector<f64, Self::NumParameters, S>) -> Self
    where
        S: Storage<f64, Self::NumParameters>;

    fn parameters(&self) -> VectorN<f64, Self::NumParameters>;

    /// Undo distortion.
    fn evaluate(&self, value: f64) -> f64;

    /// Evaluate with derivative
    fn derivative(&self, value: f64) -> f64;

    /// Evaluate with derivative
    fn with_derivative(&self, value: f64) -> (f64, f64) {
        (self.evaluate(value), self.derivative(value))
    }

    /// Apply distortion.
    fn inverse(&self, value: f64) -> f64;

    /// Parameter gradient
    fn gradient(&self, value: f64) -> VectorN<f64, Self::NumParameters>;
}
