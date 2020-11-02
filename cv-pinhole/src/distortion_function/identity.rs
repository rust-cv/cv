use super::DistortionFunction;
use cv_core::nalgebra::{
    allocator::Allocator, zero, ArrayStorage, DefaultAllocator, Dim, DimName, NamedDim, VectorN, U1,
};

/// Identity distortion, i.e. no distortion at all.
#[derive(Clone, PartialEq, Default, Debug)]
struct Identity;

impl DistortionFunction for Identity {
    type NumParameters;

    fn evaluate(&self, value: f64) -> f64 {
        value
    }

    fn derivative(&self, value: f64) -> f64 {
        0.0
    }

    fn inverse(&self, value: f64) -> f64 {
        value
    }
}
