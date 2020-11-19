use crate::distortion_function::DistortionFunction;
use crate::CameraIntrinsics;
use crate::NormalizedKeyPoint;

use cv_core::nalgebra::{allocator::Allocator, DefaultAllocator, Dim, Matrix2, Vector2};
use cv_core::{CameraModel, ImagePoint, KeyPoint};

/// Realistic camera with distortions.
///
/// Implements a realistic camera model with [optical distortions](https://en.wikipedia.org/wiki/Distortion_(optics)):
///
/// * *Radial distortion* models imperfect lens shapes.
/// * *Decentering distortion* models lenses being off-centered of the optical axis.
/// * *Thin prism distortion* models lenses not being orthogonal to optical axis.
///
/// In particular it implements the [Brown-Conrady][b71] distortion model with added thin prism
/// correction as in [Weng][w92], comparable to the model in [OpenCV][o45]. Instead of a fixed
/// number of coefficients, the camera model uses generic [`DistortionFunction`]s, provided as a type
/// parameter.
///
/// Given image coordinates $(x,y)$ and $r = x^2 + y^2$ the undistorted coordinates $(x', y')$ are
/// computed as:
///
/// $$
/// \begin{bmatrix} x' \\\\ y' \end{bmatrix} = \begin{bmatrix}
///  x ⋅ f_r(r^2, \vec θ_r) + \p{2⋅x⋅\p{t_1 ⋅ x + t_2 ⋅ y} + t_1 ⋅ r^2} ⋅ f_t(r^2, \vec θ_d) + f_{p}(r^2, \vec θ_{x})
/// \\\\
///  y ⋅ f_r(r^2, \vec θ_r) + \p{2⋅y⋅\p{t_1 ⋅ x + t_2 ⋅ y} + t_2 ⋅ r^2} ⋅ f_t(r^2, \vec θ_d) + f_{p}(r^2, \vec θ_{y})
/// \end{bmatrix}
/// $$
///
/// where $f_r$ is the radial distortion function of type `R`, $(t_1, t_2, f_t)$ specify the
/// decentering distortion, with $f_t$ of type `D` and $(f_{px}, f_{py})$ specify the thin prism
/// of type `T`.
///
/// The parameter vector of the distortion is
///
/// $$
/// \vec θ = \begin{bmatrix} \vec θ_r \\\\ t_1 \\\\ t_2 \\\\ \vec θ_d \\\\ \vec θ_x \\\\ \vec θ_y \end{bmatrix}
/// $$
///
/// # References
///
/// * D.C. Brown (1996). Decentering Distortion of Lenses. [pdf][b66].
/// * D.C. Brown (1971). Close-Range Camera Calibration. [pdf][b71].
/// * C. Stachniss (2020). Camera Parameters - Extrinsics and Intrinsics. [lecture video][s20].
/// * S. Chen, H. Jin, J. Chien, E.Chan, D. Goldman (2010). Adobe Camera Model. [pdf][a10].
/// * J. Weng (1992). Camera Calibration with Distortion Models and Accuracy Evaluation. [pdf][w92].
///
/// [b66]: https://web.archive.org/web/20180312205006/https://www.asprs.org/wp-content/uploads/pers/1966journal/may/1966_may_444-462.pdf
/// [b71]: https://www.asprs.org/wp-content/uploads/pers/1971journal/aug/1971_aug_855-866.pdf
/// [s20]: https://youtu.be/uHApDqH-8UE?t=2643
/// [a10]: http://download.macromedia.com/pub/labs/lensprofile_creator/lensprofile_creator_cameramodel.pdf
/// [w92]: https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/camera%20distortion.pdf
/// [o45]: https://docs.opencv.org/4.5.0/d9/d0c/group__calib3d.html
///
/// # To do
///
/// * Chromatic abberation.
/// * Vigneting correction.
/// * Support [Tilt/Scheimpflug](https://en.wikipedia.org/wiki/Scheimpflug_principle) lenses. See [Louhichi et al.](https://iopscience.iop.org/article/10.1088/0957-0233/18/8/037) for a mathematical model.
///
#[derive(Clone, PartialEq, PartialOrd, Debug)]
// #[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Camera<R, D, T>
where
    R: DistortionFunction,
    D: DistortionFunction,
    T: DistortionFunction,
    DefaultAllocator: Allocator<f64, R::NumParameters>,
    DefaultAllocator: Allocator<f64, D::NumParameters>,
    DefaultAllocator: Allocator<f64, T::NumParameters>,
{
    linear: CameraIntrinsics,
    radial_distortion: R,
    tangential: [f64; 2],
    tangential_distortion: D,
    prism_distortion: [T; 2],
}

impl<R, D, T> Camera<R, D, T>
where
    R: DistortionFunction,
    D: DistortionFunction,
    T: DistortionFunction,
    DefaultAllocator: Allocator<f64, R::NumParameters>,
    DefaultAllocator: Allocator<f64, D::NumParameters>,
    DefaultAllocator: Allocator<f64, T::NumParameters>,
{
    pub fn new(
        linear: CameraIntrinsics,
        radial_distortion: R,
        tangential_distortion: D,
        prism_distortion: [T; 2],
    ) -> Self {
        Self {
            linear,
            radial_distortion,
            tangential: [0.0, 0.0],
            tangential_distortion,
            prism_distortion,
        }
    }

    /// Computes $\vec f(\mathtt{point})$.
    pub fn correct(&self, point: Vector2<f64>) -> Vector2<f64> {
        let r2 = point.norm_squared();
        let f_r = self.radial_distortion.evaluate(r2);
        let f_t = 1.0;
        let f_px = self.prism_distortion[0].evaluate(r2);
        let f_py = self.prism_distortion[1].evaluate(r2);
        let t = self.tangential[0] * point.x + self.tangential[1] * point.y;
        let t_x = 2.0 * point.x * t + self.tangential[0] * r2;
        let t_y = 2.0 * point.y * t + self.tangential[1] * r2;
        Vector2::new(
            point.x * f_r + t_x * f_t + f_px,
            point.y * f_r + t_y * f_t + f_py,
        )
    }

    /// Computes $∇_{\vec x} \vec f\p{\vec x, \vec θ}$.
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

impl<R, T, P> CameraModel for Camera<R, T, P>
where
    R: DistortionFunction,
    T: DistortionFunction,
    P: DistortionFunction,
    DefaultAllocator: Allocator<f64, R::NumParameters>,
    DefaultAllocator: Allocator<f64, T::NumParameters>,
    DefaultAllocator: Allocator<f64, P::NumParameters>,
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

pub mod camera {
    use super::Camera;
    use crate::distortion_function::{Identity, Polynomial, Rational};
    use cv_core::nalgebra::{U1, U2, U3, U4};

    /// OpenCV Camera model with 4 parameters
    pub type OpenCV4 = Camera<Polynomial<U2>, Identity, Identity>;

    pub type OpenCV5 = Camera<Polynomial<U3>, Identity, Identity>;

    pub type OpenCV8 = Camera<Rational<U3, U3>, Identity, Identity>;

    pub type OpenCV12 = Camera<Rational<U4, U4>, Identity, Polynomial<U3>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distortion_function::{Identity, Polynomial, Rational};
    use cv_core::nalgebra::{Point2, Vector3, VectorN, U1, U2, U3, U4, U8};

    #[test]
    fn camera_1() {
        // Calibration parameters for a GoPro Hero 6 using OpenCV
        let focal: [f64; 2] = [1.77950719e+0, 1.77867606e+03];
        let center: [f64; 2] = [2.03258212e+03, 1.52051932e+03];
        #[rustfmt::skip]
        let distortion: [f64; 12] = [
            2.56740016e+01,  1.52496764e+01, -5.01712057e-04,
            1.09310463e-03,  6.72953083e-01,  2.59544797e+01,
            2.24213453e+01,  3.04318306e+00, -3.23278793e-03,
            9.53176056e-05, -9.35687185e-05,  2.96341863e-05];
        // The OpenCV
        let distorted = [1718.3195, 1858.1052];
        let undistorted = [-0.17996894, 0.19362147];

        let mut intrinsic = CameraIntrinsics::identity();
        intrinsic.focals = Vector2::new(focal[0], focal[1]);
        intrinsic.principal_point = Point2::new(center[0], center[1]);
        #[rustfmt::skip]
        let radial = Rational::<U4, U4>::from_parameters(
            VectorN::<f64, U8>::from_column_slice_generic(U8, U1, &[
                distortion[4], distortion[1], distortion[0], 1.0,
                distortion[7], distortion[6], distortion[5], 1.0,
            ])
        );
        let tangential = Identity;
        let prism = [
            Polynomial::<U3>::from_parameters(Vector3::new(distortion[9], distortion[8], 0.0)),
            Polynomial::<U3>::from_parameters(Vector3::new(distortion[11], distortion[10], 0.0)),
        ];

        let camera = camera::OpenCV12::new(intrinsic, radial, tangential, prism);

        let distorted = Vector2::new(1718.3195, 1858.1052);
        let undistorted = camera.correct(distorted);
        let expected = Vector2::new(-0.17996894, 0.19362147);
        println!("{:#?}", undistorted);

        let distorted = Vector2::new(579.8596, 2575.0476);
        let undistorted = camera.correct(distorted);
        let expected = Vector2::new(-1.1464612, 0.8380858);
        println!("{:#?}", undistorted);

        let source = [
            [-0.17996894, 0.19362147],
            [-0.17968875, 0.24255648],
            [-0.17944576, 0.29271868],
            [-0.17916603, 0.34418374],
            [-0.17893367, 0.39675304],
            [-0.17863423, 0.45061454],
            [-0.1782976, 0.5059651],
            [-0.17797716, 0.56267387],
            [-0.17769286, 0.62078154],
            [-0.22691241, 0.1944314],
            [-0.22716342, 0.24451172],
            [-0.22748885, 0.29591268],
            [-0.22788039, 0.3486148],
            [-0.22820546, 0.4025184],
            [-0.22855103, 0.4577957],
            [-0.22891289, 0.51458186],
            [-0.22923958, 0.572837],
            [-0.22955456, 0.6325643],
            [-0.2760499, 0.19522506],
            [-0.27696946, 0.24657097],
            [-0.27798527, 0.29921785],
            [-0.27897915, 0.35328886],
            [-0.27998206, 0.40862617],
            [-0.28099003, 0.46533334],
            [-0.28205746, 0.5236744],
            [-0.2831492, 0.5835057],
            [-0.28426304, 0.6449713],
            [-0.3276202, 0.19608246],
            [-0.3292424, 0.24866305],
            [-0.3309349, 0.3026873],
            [-0.33262616, 0.3581513],
            [-0.33433127, 0.4149485],
            [-0.33617917, 0.4732646],
            [-0.33799866, 0.5331862],
            [-0.33992943, 0.5947946],
            [-0.34184524, 0.65808254],
            [-0.38177538, 0.19692989],
            [-0.38411814, 0.25095174],
            [-0.3866268, 0.30635145],
            [-0.38909492, 0.36327213],
            [-0.3916721, 0.42171457],
            [-0.39424962, 0.48160818],
            [-0.39702195, 0.5432604],
            [-0.39976382, 0.6066253],
            [-0.40267637, 0.671886],
            [-0.43867132, 0.19787468],
            [-0.44195008, 0.2532662],
            [-0.44519594, 0.31016332],
            [-0.44858345, 0.3686112],
            [-0.45198414, 0.42872244],
            [-0.45556828, 0.49036437],
            [-0.45910528, 0.55382824],
            [-0.46299413, 0.6192026],
            [-0.466843, 0.6863503],
            [-0.49861795, 0.1987457],
            [-0.5026725, 0.2557629],
            [-0.5069558, 0.31417185],
            [-0.51119965, 0.37434888],
            [-0.5157122, 0.43618685],
            [-0.5201786, 0.49963355],
            [-0.5250113, 0.56500995],
            [-0.529766, 0.63237023],
            [-0.53485453, 0.70181614],
            [-0.56156373, 0.19966456],
            [-0.5667678, 0.25814596],
            [-0.5719744, 0.31840706],
            [-0.57744604, 0.38031867],
            [-0.5828779, 0.4440139],
            [-0.58866763, 0.50954485],
            [-0.59443516, 0.5769114],
            [-0.6006511, 0.64649254],
            [-0.60675824, 0.71801007],
            [-0.628279, 0.20040697],
            [-0.6344018, 0.26069504],
            [-0.64092815, 0.32267714],
            [-0.6474141, 0.38664073],
            [-0.6542724, 0.4522714],
            [-0.6610637, 0.51989216],
            [-0.6683662, 0.5895391],
            [-0.6756135, 0.66126966],
            [-0.6833462, 0.7354248],
            [-0.6985042, 0.20141184],
            [-0.7060343, 0.2632588],
            [-0.71366554, 0.3273413],
            [-0.7216309, 0.39327696],
            [-0.7295911, 0.46094468],
            [-0.7380902, 0.53080666],
            [-0.7465825, 0.6027301],
            [-0.7556527, 0.6770891],
            [-0.7646206, 0.75370055],
            [-0.7728095, 0.20244533],
            [-0.78165317, 0.26625928],
            [-0.7907952, 0.33220664],
            [-0.8001839, 0.40018782],
            [-0.8098858, 0.4701321],
            [-0.8196281, 0.54215276],
            [-0.83001417, 0.6167809],
            [-0.8402761, 0.6934916],
            [-0.8512289, 0.7733144],
            [-0.8518392, 0.20366336],
            [-0.862248, 0.26939756],
            [-0.8729417, 0.33734974],
            [-0.8838656, 0.40754512],
            [-0.89505607, 0.4797963],
            [-0.90671587, 0.5545365],
            [-0.9185119, 0.63145286],
            [-0.93089086, 0.71136534],
            [-0.9433173, 0.7935529],
            [-0.93565977, 0.20484428],
            [-0.9476607, 0.27270162],
            [-0.9600662, 0.3428219],
            [-0.97279584, 0.41510573],
            [-0.9859218, 0.49006754],
            [-0.9992419, 0.56712496],
            [-1.0130187, 0.6472003],
            [-1.0271673, 0.72976106],
            [-1.0417787, 0.8154801],
            [-1.02476, 0.20601285],
            [-1.038625, 0.27609447],
            [-1.0529088, 0.34840867],
            [-1.0676082, 0.42343187],
            [-1.0825946, 0.50079226],
            [-1.0980939, 0.5808948],
            [-1.1138377, 0.6636289],
            [-1.1300658, 0.7495205],
            [-1.1464612, 0.8380858],
        ];
        for i in 0..source.len() {
            let [x, y] = source[i];
            let source = Vector2::new(x, y);
            let destination = camera.correct(source);
            let (x, y) = (destination[0], destination[1]);
            let x = x * focal[0] + center[0];
            let y = y * focal[1] + center[1];
            println!("[{}, {}],", x, y);
        }
    }
}
