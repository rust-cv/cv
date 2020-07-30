use derive_more::{AsMut, AsRef, Deref, DerefMut, From, Into};
use nalgebra::{Matrix3, Matrix4, Rotation3, Unit, Vector3};
use num_traits::Float;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

/// Contains a member of the lie algebra so(3), a representation of the tangent space
/// of 3d rotation. This is also known as the lie algebra of the 3d rotation group SO(3).
///
/// This is only intended to be used in optimization problems where it is desirable to
/// have unconstranied variables representing the degrees of freedom of the rotation.
/// In all other cases, a rotation matrix should be used to store rotations, since the
/// conversion to and from a rotation matrix is non-trivial.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsMut, AsRef, Deref, DerefMut, From, Into)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
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
    /// from the exponential map of this so(3) element into SO(3). The result is converted
    /// to homogeneous form (by adding a new dimension with a `1` in the diagonal) so
    /// that it is compatible with homogeneous coordinates.
    ///
    /// If you have the rotation matrix already, please use the rotation matrix itself
    /// rather than calling this method. Calling this method will waste time converting
    /// the [`Skew3`] back into a [`Rotation3`], which is non-trivial.
    pub fn jacobian_input(self) -> Matrix4<f64> {
        let rotation: Rotation3<f64> = self.into();
        let matrix: Matrix3<f64> = rotation.into();
        matrix.to_homogeneous()
    }

    /// The jacobian of the output of a rotation in respect to the
    /// rotation itself.
    ///
    /// `y = R * x`
    ///
    /// `dy/dR = -hat(y)`
    ///
    /// The derivative is purely based on the current output vector, and thus doesn't take `self`.
    ///
    /// Note that when working with homogeneous projective coordinates, only the first three components
    /// (the bearing) are relevant, hence the resulting matrix is a [`Matrix3`].
    pub fn jacobian_self(y: Vector3<f64>) -> Matrix3<f64> {
        y.cross_matrix()
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
