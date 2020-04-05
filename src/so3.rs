use nalgebra::{Matrix3, Rotation3, Unit, Vector3};
use num_traits::Float;

/// Contains a member of the lie algebra so(3), a representation of the tangent space
/// of 3d rotation. This is also known as the lie algebra of the 3d rotation group SO(3).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Skew3(pub Vector3<f64>);

impl Skew3 {
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

impl From<Skew3> for Rotation3<f64> {
    fn from(w: Skew3) -> Self {
        let theta2 = w.0.norm_squared();
        if theta2 <= f64::epsilon() {
            Self::identity()
        } else {
            let theta = theta2.sqrt();
            let axis = Unit::new_unchecked(w.0 / theta);
            Self::from_axis_angle(&axis, theta)
        }
    }
}

impl From<Rotation3<f64>> for Skew3 {
    fn from(r: Rotation3<f64>) -> Self {
        Self(r.scaled_axis())
    }
}
