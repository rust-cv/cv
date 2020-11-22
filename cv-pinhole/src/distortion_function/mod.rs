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
    ///
    /// # Method
    ///
    /// Uses a Newton-Bisection hybrid method based on [1] that is guaranteed to
    /// converge to almost machine precision.
    ///
    /// # Resources
    ///
    /// Numerical Recipes 2nd edition. p. 365
    ///
    /// <https://github.com/osveliz/numerical-veliz/blob/master/src/rootfinding/NewtSafe.adb>
    ///
    /// # Panics
    ///
    /// Panics when an inverse does not exist in the range $x \in [0, 3]$.
    ///
    fn inverse(&self, value: f64) -> f64 {
        let (mut xl, mut xh) = (0.0, 3.0);
        let fl = self.evaluate(xl) - value;
        if fl == 0.0 {
            return xl;
        }
        let fh = self.evaluate(xh) - value;
        if fh == 0.0 {
            return xh;
        }
        if fl * fh > 0.0 {
            panic!("Inverse outside of bracket [0, 3].");
        }
        if fl > 0.0 {
            std::mem::swap(&mut xl, &mut xh);
        }
        let mut rts = 0.5 * (xl + xh);
        let mut dxold = (xl - xh).abs();
        let mut dx = dxold;
        let (mut f, mut df) = self.with_derivative(rts);
        loop {
            if (((rts - xh) * df - f) * ((rts - xl) * df - f) > 0.0)
                || (2.0 * f.abs() > (dxold * df).abs())
            {
                // Bisection step
                dxold = dx;
                dx = 0.5 * (xh - xl);
                rts = xl + dx;
                if xl == rts || xh == rts {
                    return rts;
                }
            } else {
                // Newton step
                dxold = dx;
                dx = f / df;
                let tmp = rts;
                rts -= dx;
                if tmp == rts {
                    return rts;
                }
            }
            let (nf, ndf) = self.with_derivative(rts);
            f = nf - value;
            df = ndf;
            if f < 0.0 {
                xl = rts;
            } else {
                xh = rts;
            }
        }
    }

    /// Parameter gradient $∇_{\vec β​} f(\mathtt{value}, \vec β)$.
    fn gradient(&self, value: f64) -> VectorN<f64, Self::NumParameters>;
}
