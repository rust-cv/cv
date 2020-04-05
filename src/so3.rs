use nalgebra::{Matrix3, Rotation3, Unit, Vector3};
use num_traits::Float;

/// Contains a member of the lie algebra so(3), a representation of the tangent space
/// of 3d rotation. This is also known as the lie algebra of the 3d rotation group SO(3).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Skew3(pub Vector3<f64>);

impl Skew3 {
    /// This converts a matrix in skew-symmetric form into a Skew3.
    pub fn vee(mat: Matrix3<f64>) -> Self {
        Self(Vector3::new(mat.m32, mat.m13, mat.m21))
    }

    /// This converts the Skew3 into its skew-symmetric matrix form.
    #[rustfmt::skip]
    pub fn hat(self) -> Matrix3<f64> {
        let w = self.0;
        Matrix3::new(
            0.0,  -w.z,   w.y,
            w.z,   0.0,  -w.x,
           -w.y,   w.x,   0.0,
       )
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
