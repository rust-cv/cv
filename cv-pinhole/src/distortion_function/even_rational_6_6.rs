use super::DistortionFunction;
use num_traits::Float;

/// Maxmimum number of iterations to use in Newton-Raphson inversion.
const MAX_ITERATIONS: usize = 100;

/// Convergence treshold for Newton-Raphson inversion.
const EPSILON: f64 = f64::EPSILON;

/// Even $(6,6)$-rational distortion function
///
/// The radial distortion is calibrated by scaling with a degree $(6,6)$ [even](https://en.wikipedia.org/wiki/Even_and_odd_functions#Even_functions) [rational](https://en.wikipedia.org/wiki/Rational_function) function:
///
/// $$
/// r' = r ⋅ \frac{1 + k_1 ⋅ r^2 + k_2 ⋅ r^4 + k_3 ⋅ r^6}{1 + k_4 ⋅ r^2 + k_5 ⋅ r^4 + k_6 ⋅ r^6}
/// $$
///
#[derive(Clone, PartialEq, Default, Debug)]
pub struct EvenRational66(pub [f64; 6]);

impl EvenRational66 {}

impl DistortionFunction for EvenRational66 {
    /// Given $r$ compute $r'$.
    #[rustfmt::skip]
    fn evaluate(&self, r: f64) -> f64 {
        let r2 = r * r;
        let mut p = self.0[2];
        p *= r2; p += self.0[1];
        p *= r2; p += self.0[0];
        p *= r2; p += 1.0;
        let mut q = self.0[5];
        q *= r2; q += self.0[4];
        q *= r2; q += self.0[3];
        q *= r2; q += 1.0;
        r * p / q
    }

    /// Given $r'$ compute $r$.
    ///
    /// # Examples
    ///
    /// ```
    /// use cv_pinhole::RadialDistortion;
    ///
    /// let distortion = RadialDistortion([0.1, 0.2, 0.3, 0.4, 0.2, 0.1]);
    /// let camera_r = 1.3;
    /// let corrected_r = distortion.calibrate(camera_r);
    /// let reprojected_r = distortion.uncalibrate(corrected_r);
    /// assert_eq!(camera_r, reprojected_r);
    /// ```
    ///
    /// # Method
    ///
    /// Given $r'$ solve for $r$. Start with the known relation
    ///
    ///
    /// $$
    /// r' = r ⋅ \frac{1 + k_1 ⋅ r^2 + k_2 ⋅ r^4 + k_3 ⋅ r^6}
    /// {1 + k_4 ⋅ r^2 + k_5 ⋅ r^4 + k_6 ⋅ r^6}
    /// $$
    ///
    /// manipulate the rational relation into a polynomial
    ///
    /// $$
    /// \begin{aligned}
    /// r' ⋅ \p{1 + k_4 ⋅ r^2 + k_5 ⋅ r^4 + k_6 ⋅ r^6}
    /// &= r ⋅ \p{1 + k_1 ⋅ r^2 + k_2 ⋅ r^4 + k_3 ⋅ r^6} \\\\
    /// r' + r' ⋅ k_4 ⋅ r^2 + r' ⋅ k_5 ⋅ r^4 + r' ⋅ k_6 ⋅ r^6
    /// &= r + k_1 ⋅ r^3 + k_2 ⋅ r^5 + k_3 ⋅ r^7 \\\\
    /// \end{aligned}
    /// $$
    ///
    /// $$
    /// r' + r' ⋅ k_4 ⋅ r^2 + r' ⋅ k_5 ⋅ r^4 + r' ⋅ k_6 ⋅ r^6
    /// \- r - k_1 ⋅ r^3 - k_2 ⋅ r^5 - k_3 ⋅ r^7 = 0
    /// $$
    ///
    /// That is, we need to find the root of the 7th-order polynomial with coefficients:
    ///
    /// $$
    /// P(r) =
    /// \begin{bmatrix}
    /// r' & -1 & r' ⋅ k_4 & -k_1 & r' ⋅ k_5 & -k_2 & r' ⋅ k_6 & - k_3
    /// \end{bmatrix}^T ⋅ V(r)
    /// $$
    ///
    /// where $V(r) = \begin{bmatrix} 1 & r & r^2 & ᠁ \end{bmatrix}$ is the [Vandermonde matrix](https://en.wikipedia.org/wiki/Vandermonde_matrix). The coefficients of the 6ht-order derivative polynomials are:
    ///
    /// $$
    /// P'(r) =
    /// \begin{bmatrix}
    /// -1 & 2 ⋅ r' ⋅ k_4 & -3 ⋅ k_1 & 4 ⋅ r' ⋅ k_5 & -5 ⋅ k_2 & 6 ⋅ r' ⋅ k_6 & - 7 ⋅ k_3
    /// \end{bmatrix}^T ⋅ V(r)
    /// $$
    ///
    /// Using $r'$ as the starting value we can approximate $r'$ using [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method):
    ///
    /// $$
    /// \begin{aligned}
    /// r_0 &= r' & r_{i+1} = r_i - \frac{P(r_i)}{P'(r_i)}
    /// \end{aligned}
    /// $$
    ///
    /// **Note.** We could also produce higher derivatives and use [Halley's method](https://en.wikipedia.org/wiki/Halley%27s_method) or higher order [Housholder methods](https://en.wikipedia.org/wiki/Householder%27s_method).
    ///
    /// The inversion is approximated using at most [`MAX_ITERATIONS`] rounds of Newton-Raphson
    /// or when the $\abs{r_{i+1} - r_i}$ is less than [`EPSILON`] times $r_{i+1}$.
    #[rustfmt::skip]
    fn inverse(&self, mut r: f64) -> f64 {
        let p = [
            r,             -1.0,
            r * self.0[3], -self.0[0],
            r * self.0[4], -self.0[1],
            r * self.0[5], -self.0[2],
        ];
        // Iterate Newtons method
        for _ in 0..MAX_ITERATIONS {
            let mut value = p[7];
            value *= r; value += p[6];
            value *= r; value += p[5];
            value *= r; value += p[4];
            value *= r; value += p[3];
            value *= r; value += p[2];
            value *= r; value += p[1];
            value *= r; value += p[0];
            let mut deriv = p[7] * 7.0;
            deriv *= r; deriv += p[6] * 6.0;
            deriv *= r; deriv += p[5] * 5.0;
            deriv *= r; deriv += p[4] * 4.0;
            deriv *= r; deriv += p[3] * 3.0;
            deriv *= r; deriv += p[2] * 2.0;
            deriv *= r; deriv += p[1];
            let delta = value / deriv;
            r -= delta;
            if Float::abs(delta) <= EPSILON * r {
                break;
            }
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invert() {
        let distortion = RadialDistortion([0.1, 0.2, 0.3, 0.4, 0.2, 0.1]);
        let test_values = [0.0, 0.5, 1.0, 1.5];
        for &camera_r in &test_values {
            let corrected = distortion.calibrate(camera_r);
            let reprojected = distortion.uncalibrate(corrected);
            assert_eq!(camera_r, reprojected);
        }
    }
}
