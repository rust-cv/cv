// use alga::{
//     general::{Id, Identity, Multiplicative},
//     linear::{
//         AffineTransformation, DirectIsometry, Isometry, OrthogonalTransformation,
//         ProjectiveTransformation, Rotation, Similarity,
//     },
// };
use core::ops::{Mul, MulAssign};
use nalgebra::{dimension::U3, AbstractRotation, Matrix3, Point3, Rotation3, Unit, Vector3};
use num_traits::Float;

/// Contains a member of the lie algebra so(3), a representation of the tangent space
/// of 3d rotation. This is also known as the lie algebra of the 3d rotation group SO(3).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Skew3(pub Vector3<f64>);

impl Skew3 {
    /// Converts the Skew3 to a Rotation3 matrix.
    pub fn rotation(self) -> Rotation3<f64> {
        self.into()
    }

    /// This converts a matrix in skew-symmetric form into a Skew3.
    ///
    /// Warning: Does no check to ensure matrix is actually skew-symmetric.
    pub fn vee(mat: Matrix3<f64>) -> Self {
        Self(Vector3::new(mat.m32, mat.m13, mat.m21))
    }

    /// This converts the Skew3 into its skew-symmetric matrix form.
    #[rustfmt::skip]
    pub fn hat(self) -> Matrix3<f64> {
        self.0.cross_matrix()
    }

    /// This converts the Skew3 into its squared skew-symmetric matrix form efficiently.
    #[rustfmt::skip]
    pub fn hat2(self) -> Matrix3<f64> {
        let w = self.0;
        let w11 = w.x * w.x;
        let w12 = w.x * w.y;
        let w13 = w.x * w.z;
        let w22 = w.y * w.y;
        let w23 = w.y * w.z;
        let w33 = w.z * w.z;
        Matrix3::new(
            -w22 - w33,     w12,           w13,
             w12,          -w11 - w33,     w23,
             w13,           w23,          -w11 - w22,
        )
    }

    /// Computes the lie bracket [self, rhs].
    pub fn bracket(self, rhs: Self) -> Self {
        Self::vee(self.hat() * rhs.hat() - rhs.hat() * self.hat())
    }

    /// The jacobian of the output of a rotation in respect to the
    /// input of a rotation.
    ///
    /// `y = R * x`
    ///
    /// `dy/dx = R`
    ///
    /// The formula is pretty simple and is just the rotation matrix created
    /// from the exponential map of this so(3) element into SO(3).
    pub fn jacobian_output_to_input(self) -> Matrix3<f64> {
        let rotation: Rotation3<f64> = self.into();
        rotation.into()
    }

    /// The jacobian of the output of a rotation in respect to the
    /// rotation itself.
    ///
    /// `y = R * x`
    ///
    /// `dy/dR = -hat(y)`
    ///
    /// The derivative is purely based on the current output vector, and thus doesn't take `self`.
    pub fn jacobian_output_to_self(y: Vector3<f64>) -> Matrix3<f64> {
        -y.cross_matrix()
    }
}

/// This is the exponential map.
impl From<Skew3> for Rotation3<f64> {
    fn from(w: Skew3) -> Self {
        // This check is done to avoid the degenerate case where the angle is near zero.
        let theta2 = w.0.norm_squared();
        if theta2 <= f64::epsilon() {
            Rotation3::from_matrix(&(Matrix3::identity() + w.hat()))
        } else {
            let theta = theta2.sqrt();
            let axis = Unit::new_unchecked(w.0 / theta);
            Self::from_axis_angle(&axis, theta)
        }
    }
}

/// This is the log map.
impl From<Rotation3<f64>> for Skew3 {
    fn from(r: Rotation3<f64>) -> Self {
        Self(r.scaled_axis())
    }
}

impl Mul for Skew3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        (self.rotation() * rhs.rotation()).into()
    }
}

impl MulAssign for Skew3 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl AbstractRotation<f64, U3> for Skew3 {
    #[inline]
    fn identity() -> Self {
        Self(Vector3::zeros())
    }

    #[inline]
    fn inverse(&self) -> Self {
        Self(-self.0)
    }

    #[inline]
    fn inverse_mut(&mut self) {
        self.0 = -self.0;
    }

    #[inline]
    fn transform_vector(&self, v: &Vector3<f64>) -> Vector3<f64> {
        self.rotation().transform_vector(v)
    }

    #[inline]
    fn transform_point(&self, p: &Point3<f64>) -> Point3<f64> {
        self.rotation().transform_point(p)
    }

    #[inline]
    fn inverse_transform_vector(&self, v: &Vector3<f64>) -> Vector3<f64> {
        self.inverse().rotation().transform_vector(v)
    }

    #[inline]
    fn inverse_transform_point(&self, p: &Point3<f64>) -> Point3<f64> {
        self.inverse().rotation().inverse_transform_point(p)
    }
}
