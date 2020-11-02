use super::DistortionFunction;
use cv_core::nalgebra::DimAdd;
use cv_core::nalgebra::DimSum;

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

    fn inverse(&self, value: f64) -> f64 {
        todo!()
    }
}
