use super::DistortionFunction;
use cv_core::nalgebra::{
    allocator::Allocator, storage::Storage, zero, ArrayStorage, DefaultAllocator, Dim, DimName,
    NamedDim, Vector, VectorN, U0, U1,
};

/// Identity distortion, i.e. no distortion at all.
///
/// $$
/// f(x) = x
/// $$
///
/// Refresh, maybe?
#[derive(Clone, PartialEq, Default, Debug)]
pub struct Identity;

impl DistortionFunction for Identity {
    type NumParameters = U0;

    fn from_parameters<S>(parameters: Vector<f64, Self::NumParameters, S>) -> Self
    where
        S: Storage<f64, Self::NumParameters>,
    {
        Self
    }

    fn parameters(&self) -> VectorN<f64, Self::NumParameters> {
        VectorN::zeros_generic(U0, U1)
    }

    fn evaluate(&self, value: f64) -> f64 {
        value
    }

    fn derivative(&self, value: f64) -> f64 {
        0.0
    }

    fn inverse(&self, value: f64) -> f64 {
        value
    }

    fn gradient(&self, value: f64) -> VectorN<f64, Self::NumParameters> {
        todo!()
    }
}
