use crate::CameraIntrinsics;
use crate::DistortionFunction;
use crate::NormalizedKeyPoint;

use cv_core::nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Matrix2, Vector2};
use cv_core::{CameraModel, ImagePoint, KeyPoint};

/// Camera with distortion.
///
/// The distortion model is compatible with OpenCV.
///
/// Given image coordinates $(x,y)$ and $r = x^2 + y^2$ the undistorted coordinates $(x', y')$ are:
///
/// $$
/// \begin{bmatrix} x' \\\\ y' \end{bmatrix} =
/// \vec f \p{\begin{bmatrix} x \\\\ y \end{bmatrix}} = \begin{bmatrix}
///  x ⋅ f_r(r^2) + \p{2⋅x⋅\p{t_1 ⋅ x + t_2 ⋅ y} + t_1 ⋅ r^2} ⋅ f_t(r^2) + f_{px}(r^2)
/// \\\\[.8em]
///  y ⋅ f_r(r^2) + \p{2⋅y⋅\p{t_1 ⋅ x + t_2 ⋅ y} + t_2 ⋅ r^2} ⋅ f_t(r^2) + f_{py}(r^2)
/// \end{bmatrix}
/// $$
///
/// where $f_r$ is the *radial* distortion function, $(t_1, t_2, f_t)$ specify the *tangential* distortion (See [Brown][b66] p. 454) and $(f_{px}, f_{py})$ the *thin prism* distortion.
///
/// **Note.** The tangential distortion compensates for lenses not being entirely parallel with the
/// image plane. Despite what the name may suggest, its contribution is not orthogonal to the
/// radial distortion.
///
/// # References
///
/// * D.C. Brown (1996). Decentering Distortion of Lenses. [online][b66].
///
/// [b66]: https://web.archive.org/web/20180312205006/https://www.asprs.org/wp-content/uploads/pers/1966journal/may/1966_may_444-462.pdf
///
///
/// https://medium.com/analytics-vidhya/camera-calibration-with-opencv-f324679c6eb7
///
/// https://spectrogrism.readthedocs.io/en/latest/distortions.html
///
///
#[derive(Clone, PartialEq, PartialOrd, Debug)]
// #[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
struct Camera<R, T, P, D1, D2, D3>
where
    R: DistortionFunction<NumParameters = D1>,
    T: DistortionFunction<NumParameters = D2>,
    P: DistortionFunction<NumParameters = D3>,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    DefaultAllocator: Allocator<f64, D1>,
    DefaultAllocator: Allocator<f64, D2>,
    DefaultAllocator: Allocator<f64, D3>,
{
    linear: CameraIntrinsics,
    radial_distortion: R,
    tangential: [f64; 2],
    tangential_distortion: T,
    prism_distortion: [P; 2],
}

impl<R, T, P, D1, D2, D3> Camera<R, T, P, D1, D2, D3>
where
    R: DistortionFunction<NumParameters = D1>,
    T: DistortionFunction<NumParameters = D2>,
    P: DistortionFunction<NumParameters = D3>,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    DefaultAllocator: Allocator<f64, D1>,
    DefaultAllocator: Allocator<f64, D2>,
    DefaultAllocator: Allocator<f64, D3>,
{
    pub fn new(linear: CameraIntrinsics, radial_distortion: R, tangential_distortion: T) -> Self {
        todo!()
        // Self {
        //     linear,
        //     radial_distortion,
        //     tangential_distortion,
        // }
    }

    /// Computes $\vec f(\mathtt{point})$.
    pub fn correct(&self, point: Vector2<f64>) -> Vector2<f64> {
        let r2 = point.norm_squared();
        let f_r = self.radial_distortion.evaluate(r2);
        let f_t = self.tangential_distortion.evaluate(r2);
        let f_px = self.prism_distortion[0].evaluate(r2) - 1.0;
        let f_py = self.prism_distortion[1].evaluate(r2) - 1.0;
        let t = self.tangential[0] * point.x + self.tangential[1] * point.y;
        let t_x = 2.0 * point.x * t + self.tangential[0] * r2;
        let t_y = 2.0 * point.y * t + self.tangential[1] * r2;
        Vector2::new(
            point.x * f_r + t_x * f_t + f_px,
            point.y * f_r + t_y * f_t + f_py,
        )
    }

    /// Computes $\mathbf{J}_{\vec f}(\mathtt{point})$.
    ///
    /// $$
    /// \mathbf{J}_{\vec f} = \begin{bmatrix}
    /// \frac{\partial f_x}{\partial x} & \frac{\partial f_x}{\partial y} \\\\[.8em]
    /// \frac{\partial f_y}{\partial x} & \frac{\partial f_y}{\partial y}
    /// \end{bmatrix}
    /// $$
    ///
    /// $$
    /// \begin{aligned}
    /// \frac{\partial f_x}{\partial x} &=
    /// f_r(r^2) + 2 ⋅ x^2 ⋅ f_r'(r^2) \\\\ &\phantom{=}
    /// + \p{6⋅t_1 ⋅ x + 2 ⋅ t_2 ⋅ y} ⋅ f_t(r^2) \\\\ &\phantom{=}
    /// + \p{2⋅x⋅\p{t_1 ⋅ x + t_2 ⋅ y} + t_1 ⋅ r^2} ⋅ 2 ⋅ x ⋅ f_t'(r^2) \\\\ &\phantom{=}
    /// + 2 ⋅ x ⋅ f_{px}'(r^2)
    /// \end{aligned}
    /// $$
    ///
    /// $$
    /// \begin{aligned}
    /// \frac{\partial f_x}{\partial y} &=
    /// 2 ⋅ x ⋅ y^2 ⋅ f_r'(r^2) \\\\ &\phantom{=}
    /// + \p{2⋅t_1 ⋅ y + 2 ⋅ t_2 ⋅ x} ⋅ f_t(r^2) \\\\ &\phantom{=}
    /// + \p{2⋅x⋅\p{t_1 ⋅ x + t_2 ⋅ y} + t_1 ⋅ r^2} ⋅ 2 ⋅ y ⋅ f_t'(r^2) \\\\ &\phantom{=}
    /// + 2 ⋅ y ⋅ f_{px}'(r^2)
    /// \end{aligned}
    /// $$
    ///
    pub fn jacobian(&self, point: Vector2<f64>) -> Matrix2<f64> {
        let r2 = point.norm_squared();
        let (f_r, df_r) = self.radial_distortion.with_derivative(r2);
        let (f_t, df_t) = self.tangential_distortion.with_derivative(r2);
        let df_px = self.prism_distortion[0].derivative(r2);
        let df_py = self.prism_distortion[1].derivative(r2);

        todo!()
        // Matrix2::new()
    }
}

impl<R, T, P, D1, D2, D3> CameraModel for Camera<R, T, P, D1, D2, D3>
where
    R: DistortionFunction<NumParameters = D1>,
    T: DistortionFunction<NumParameters = D2>,
    P: DistortionFunction<NumParameters = D3>,
    D1: Dim,
    D2: Dim,
    D3: Dim,
    DefaultAllocator: Allocator<f64, D1>,
    DefaultAllocator: Allocator<f64, D2>,
    DefaultAllocator: Allocator<f64, D3>,
{
    type Projection = NormalizedKeyPoint;

    fn calibrate<Point: ImagePoint>(&self, point: Point) -> Self::Projection {
        let NormalizedKeyPoint(distorted) = self.linear.calibrate(point);
        let distorted_r = distorted.coords.norm();
        let corrected_r = self.radial_distortion.evaluate(distorted_r);
        let r_factor = corrected_r / distorted_r;
        let corrected = (distorted.coords * r_factor).into();

        NormalizedKeyPoint(corrected)
    }

    fn uncalibrate(&self, projection: Self::Projection) -> KeyPoint {
        let NormalizedKeyPoint(corrected) = projection;
        let corrected_r = corrected.coords.norm();
        let distorted_r = self.radial_distortion.inverse(corrected_r);
        let r_factor = distorted_r / corrected_r;
        let distorted = NormalizedKeyPoint((corrected.coords * r_factor).into());
        self.linear.uncalibrate(distorted).into()
    }
}
