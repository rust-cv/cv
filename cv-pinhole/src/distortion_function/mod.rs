mod polynomial;
mod rational;

use cv_core::nalgebra::Dim;

pub trait DistortionFunction {
    /// Type level number of parameters
    type NumParameters: Dim;

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

    // TODO: Parameters, derivatives, Jacobians, etc for calibration
    // fn parmaeters(&self) -> VectorN<f64, Self::NumParameters>;
}
