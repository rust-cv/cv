use super::DistortionFunction;
use cv_core::nalgebra::{
    allocator::Allocator, zero, ArrayStorage, DefaultAllocator, Dim, DimName, NamedDim, VectorN, U1,
};

#[derive(Clone, Debug)]
struct Polynomial<Degree>(VectorN<f64, Degree>)
where
    Degree: Dim,
    DefaultAllocator: Allocator<f64, Degree>;

impl<Degree> DistortionFunction for Polynomial<Degree>
where
    Degree: Dim,
    DefaultAllocator: Allocator<f64, Degree>,
{
    type NumParameters = Degree;

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

    fn inverse(&self, value: f64) -> f64 {
        todo!()
    }
}
