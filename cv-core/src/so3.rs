use core::{
    iter::Sum,
    ops::{Add, AddAssign},
};
use derive_more::{AsMut, AsRef, Deref, DerefMut, From, Into};
use nalgebra::{Const, IsometryMatrix3, Matrix3, Matrix4, Rotation3, Unit, Vector3, Vector6};
use num_traits::Float;
#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

/// Contains a small gradient translation and rotation that will be appended to
/// the reference frame of some pose.
///
/// This is a member of the lie algebra se(3).
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Se3TangentSpace {
    pub translation: Vector3<f64>,
    pub rotation: Vector3<f64>,
}

impl Se3TangentSpace {
    #[inline(always)]
    pub fn new(mut translation: Vector3<f64>, mut rotation: Vector3<f64>) -> Self {
        if translation.iter().any(|n| n.is_nan()) {
            translation = Vector3::zeros();
        }
        if rotation.iter().any(|n| n.is_nan()) {
            rotation = Vector3::zeros();
        }
        Self {
            translation,
            rotation,
        }
    }

    #[inline(always)]
    pub fn identity() -> Self {
        Self {
            translation: Vector3::zeros(),
            rotation: Vector3::zeros(),
        }
    }

    /// Inverts the transformation, which is very cheap.
    #[must_use]
    #[inline(always)]
    pub fn inverse(self) -> Self {
        Self {
            translation: -self.translation,
            rotation: -self.rotation,
        }
    }

    /// Gets the isometry that represents this tangent space transformation.
    #[must_use]
    #[inline(always)]
    pub fn isometry(self) -> IsometryMatrix3<f64> {
        let rotation = Rotation3::from_scaled_axis(self.rotation);
        IsometryMatrix3::from_parts((rotation * self.translation).into(), rotation)
    }

    /// For tangent spaces where the translation and rotation are both rotational, this retrieves the
    /// translation and rotation rotation matrix. The rotation matrix for the translation rotates the translation,
    /// while the rotation matrix for the rotation is left-multiplied by the rotation.
    ///
    /// Returns `(translation_rotation, rotation)`.
    #[must_use]
    #[inline(always)]
    pub fn rotations(self) -> (Rotation3<f64>, Rotation3<f64>) {
        let translation_rotation = Rotation3::from_scaled_axis(self.translation);
        let rotation = Rotation3::from_scaled_axis(self.rotation);
        (translation_rotation, rotation)
    }

    /// Scales both the rotation and the translation.
    #[must_use]
    #[inline(always)]
    pub fn scale(mut self, scale: f64) -> Self {
        self.translation *= scale;
        self.rotation *= scale;
        self
    }

    /// Scales the translation.
    #[must_use]
    #[inline(always)]
    pub fn scale_translation(mut self, scale: f64) -> Self {
        self.translation *= scale;
        self
    }

    /// Scales the rotation.
    #[must_use]
    #[inline(always)]
    pub fn scale_rotation(mut self, scale: f64) -> Self {
        self.rotation *= scale;
        self
    }

    #[inline(always)]
    pub fn to_vec(&self) -> Vector6<f64> {
        Vector6::new(
            self.translation.x,
            self.translation.y,
            self.translation.z,
            self.rotation.x,
            self.rotation.y,
            self.rotation.z,
        )
    }

    #[inline(always)]
    pub fn from_vec(v: Vector6<f64>) -> Self {
        Self {
            translation: v.rows_generic(0, Const::<3>).into_owned(),
            rotation: v.rows_generic(3, Const::<3>).into_owned(),
        }
    }

    /// Assumes an L2 tangent space is provided as input and returns the L1 tangent space.
    #[inline(always)]
    #[must_use]
    pub fn l1(&self) -> Self {
        Self::new(self.translation.normalize(), self.rotation.normalize())
    }
}

impl Add for Se3TangentSpace {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            translation: self.translation + rhs.translation,
            rotation: self.rotation + rhs.rotation,
        }
    }
}

impl AddAssign for Se3TangentSpace {
    fn add_assign(&mut self, rhs: Self) {
        self.translation += rhs.translation;
        self.rotation += rhs.rotation;
    }
}

impl Sum for Se3TangentSpace {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Se3TangentSpace::identity(), |a, b| a + b)
    }
}

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

    /// Converts the Skew3 into a Rotation3 matrix quickly, but only works when the rotation
    /// is very small.
    pub fn rotation_small(self) -> Rotation3<f64> {
        Rotation3::from_matrix(&(Matrix3::identity() + self.hat()))
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
    #[must_use]
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
            w.rotation_small()
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
        let skew3 = r.scaled_axis();
        // TODO: File issue on `nalgebra`, as this shouldn't happen and is bug.
        let skew3 = if skew3.iter().any(|n| n.is_nan()) {
            Vector3::zeros()
        } else {
            skew3
        };
        Self(skew3)
    }
}
