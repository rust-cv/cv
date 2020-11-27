use crate::distortion_function::{DistortionFunction, Fisheye};
use crate::CameraIntrinsics;
use crate::NormalizedKeyPoint;
use crate::{newton2, root};

use cv_core::nalgebra::{
    allocator::Allocator, DefaultAllocator, DimAdd, DimMul, DimName, DimProd, DimSum, Matrix2,
    Point2, Vector2, VectorN, U1, U2,
};
use cv_core::{CameraModel, ImagePoint, KeyPoint};
use num_traits::Zero;

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
/// ## From model coordinates to image coordinates
///
/// Given a point $(x, y, z)$ in camera relative coordinates such
/// that the camera is position at the origin and looking in the +z-direction.
///
/// Given undistorted image coordinates $(x,y)$ and $r = x^2 + y^2$ the distorted coordinates $(x', y')$ are
/// computed as:
///
/// $$
/// z' ⋅ \begin{bmatrix} x' \\\\ y' \\\ 1 \end{bmatrix} =
/// \begin{bmatrix}
///    f_x & s & c_x \\\\
///    0 & f_y & c_y \\\\
///    0 & 0 & 1
/// \end{bmatrix}
/// \begin{bmatrix} x \\\\ y \\\\ z \end{bmatrix}
/// $$
///
/// $$
/// \begin{bmatrix} x' \\\\ y' \end{bmatrix} =
/// \frac{f_p(r, \vec θ_p)}{r} ⋅ \begin{bmatrix} x \\\\ y \end{bmatrix}
/// $$
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
/// ## From image coordinates to model coordinates
///
/// ## Parameterization
///
/// The parameter vector of the distortion is
///
/// $$
/// \vec θ = \begin{bmatrix} k & \vec θ_r^T & t_1 & t_2 & \vec θ_d^T & \vec θ_x^T & \vec θ_y^T \end{bmatrix}^T
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
/// * Replace all the [`Dim`] madness with const generics once this hits stable.
/// * Adobe model fisheye distortion, chromatic abberation and vigneting correction.
/// * Support [Tilt/Scheimpflug](https://en.wikipedia.org/wiki/Scheimpflug_principle) lenses. See [Louhichi et al.](https://iopscience.iop.org/article/10.1088/0957-0233/18/8/037) for a mathematical model.
///
#[derive(Clone, PartialEq, PartialOrd, Default, Debug)]
// #[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct Camera<R, T, P>
where
    R: DistortionFunction,
    T: DistortionFunction,
    P: DistortionFunction,
    DefaultAllocator: Allocator<f64, R::NumParameters>,
    DefaultAllocator: Allocator<f64, T::NumParameters>,
    DefaultAllocator: Allocator<f64, P::NumParameters>,
{
    linear: CameraIntrinsics,
    fisheye: Fisheye,
    radial_distortion: R,
    tangential: [f64; 2],
    tangential_distortion: T,
    prism_distortion: [P; 2],
}

impl<R, T, P> Camera<R, T, P>
where
    R: DistortionFunction,
    T: DistortionFunction,
    P: DistortionFunction,
    DefaultAllocator: Allocator<f64, R::NumParameters>,
    DefaultAllocator: Allocator<f64, T::NumParameters>,
    DefaultAllocator: Allocator<f64, P::NumParameters>,
{
    /// Apply lens distortions to a point.
    #[rustfmt::skip]
    pub fn distort(&self, point: Point2<f64>) -> Point2<f64> {
        let r2 = point.coords.norm_squared();
        let f_r = self.radial_distortion.evaluate(r2);
        let f_t = self.tangential_distortion.evaluate(r2);
        let f_px = self.prism_distortion[0].evaluate(r2);
        let f_py = self.prism_distortion[1].evaluate(r2);
        let [p_1, p_2] = self.tangential;
        let (x, y) = (point.x, point.y);
        let t_x = 2.0 * p_1 * x * y + p_2 * (r2 + 2.0 * x * x);
        let t_y = p_1 * (r2 + 2.0 * y * y) + 2.0 * p_2 * x * y;
        Point2::new(
            x * f_r + t_x * f_t + f_px,
            y * f_r + t_y * f_t + f_py,
        )
    }

    /// Apply lens distortions to a point.
    #[rustfmt::skip]
    pub fn undistort(&self, point: Point2<f64>) -> Point2<f64> {
        // The radial distortion is a large effect. It is also a one-dimensional
        // problem we can solve precisely. This will produce a good starting
        // point for the two-dimensional problem.
        let rd = point.coords.norm();
        // Find $r_u$ such that $r_d = r_u ⋅ f(r_u^2)$
        let ru = root(|ru| {
            let (f, df) = self.radial_distortion.with_derivative(ru * ru);
            (ru * f - rd, f + 2.0 * ru * ru * df)
        }, 0.0, 10.0);
        let pu = point * ru / rd;
        
        // Newton-Raphson iteration
        let pu = newton2(|x| {
            let (f, j) = self.with_jacobian(x.into());
            (f.coords - point.coords, j)
        }, pu.coords).unwrap();
        pu.into()
    }

    /// Convert from rectilinear to the camera projection.
    pub fn project(&self, point: Point2<f64>) -> Point2<f64> {
        let r = point.coords.norm();
        let rp = self.fisheye.evaluate(r);
        point * rp / r
    }

    /// Convert from camera projection to rectilinear.
    pub fn unproject(&self, point: Point2<f64>) -> Point2<f64> {
        let rp = point.coords.norm();
        let r = self.fisheye.inverse(rp);
        point * r / rp
    }

    /// Computes $\p{f\p{\vec x, \vec θ}, ∇_{\vec x} \vec f\p{\vec x, \vec θ}}$.
    ///
    /// $$
    /// ∇_{\vec x} \vec f\p{\vec x, \vec θ} = \begin{bmatrix}
    /// \frac{\partial f_x}{\partial x} & \frac{\partial f_x}{\partial y} \\\\[.8em]
    /// \frac{\partial f_y}{\partial x} & \frac{\partial f_y}{\partial y}
    /// \end{bmatrix}
    /// $$
    ///
    pub fn with_jacobian(&self, point: Point2<f64>) -> (Point2<f64>, Matrix2<f64>) {
        let [x, y] = [point.coords[0], point.coords[1]];
        let r2 = x * x + y * y;
        let r2dx = 2. * x;
        let r2dy = 2. * y;
        let (f_r, df_r) = self.radial_distortion.with_derivative(r2);
        let (f_t, df_t) = self.tangential_distortion.with_derivative(r2);
        let (f_px, df_px) = self.prism_distortion[0].with_derivative(r2);
        let (f_py, df_py) = self.prism_distortion[1].with_derivative(r2);
        let [t1, t2] = self.tangential;
        let tx = 2.0 * t1 * x * y + t2 * (r2 + 2.0 * x * x);
        let txdx = 2.0 * t1 * y + t2 * (r2dx + 4.0 * x);
        let txdy = 2.0 * t1 * x + t2 * r2dy;
        let ty = t1 * (r2 + 2.0 * y * y) + 2.0 * t2 * x * y;
        let tydx = t1 * r2dx + 2.0 * t2 * y;
        let tydy = t1 * (r2dy + 4.0 * y) + 2.0 * t2 * x;
        let u = x * f_r + tx * f_t + f_px;
        let udr2 = x * df_r + tx * df_t + df_px;
        let udx = f_r + txdx * f_t + udr2 * r2dx;
        let udy = txdy * f_t + udr2 * r2dy;
        let v = y * f_r + ty * f_t + f_py;
        let vdr2 = y * df_r + ty * df_t + df_py;
        let vdx = tydx * f_t + vdr2 * r2dx;
        let vdy = f_r + tydy * f_t + vdr2 * r2dy;
        (Point2::new(u, v), Matrix2::new(udx, udy, vdx, vdy))
    }

    /// Computes $∇_{\vec θ} \vec f\p{\vec x, \vec θ}$
    pub fn parameter_jacobian(&self, point: Point2<f64>) -> (Point2<f64>, Matrix2<f64>) {
        todo!()
    }
}

type Dim<P>
where
    P: DistortionFunction,
    P::NumParameters: DimMul<U2>,
    DimProd<P::NumParameters, U2>: DimAdd<U2>,
= DimSum<DimProd<P::NumParameters, U2>, U2>;

impl<R, T, P> Camera<R, T, P>
where
    R: DistortionFunction,
    T: DistortionFunction,
    P: DistortionFunction,
    DefaultAllocator: Allocator<f64, R::NumParameters>,
    DefaultAllocator: Allocator<f64, T::NumParameters>,
    DefaultAllocator: Allocator<f64, P::NumParameters>,
    R::NumParameters: DimName,
    T::NumParameters: DimName,
    P::NumParameters: DimName + DimMul<U2>,
    DimProd<P::NumParameters, U2>: DimName + DimAdd<U2>,
    Dim<P>: DimName,
    DefaultAllocator: Allocator<f64, Dim<P>>,
{
    pub fn parameters(&self) -> VectorN<f64, Dim<P>> {
        let fisheye_params = self.fisheye.parameters();
        let radial_params = self.radial_distortion.parameters();
        let tangential_params = self.tangential_distortion.parameters();
        let prism_params_x = self.prism_distortion[0].parameters();
        let prism_params_y = self.prism_distortion[1].parameters();

        let mut result = VectorN::<f64, Dim<P>>::zero();
        let mut offset = 0;
        result
            .fixed_slice_mut::<U1, U1>(offset, 0)
            .copy_from(&fisheye_params);
        offset += fisheye_params.nrows();
        result
            .fixed_slice_mut::<R::NumParameters, U1>(offset, 0)
            .copy_from(&radial_params);
        offset += radial_params.nrows();
        result[(offset, 0)] = self.tangential[0];
        offset += 1;
        result[(offset, 0)] = self.tangential[1];
        offset += 1;
        result
            .fixed_slice_mut::<T::NumParameters, U1>(offset, 0)
            .copy_from(&tangential_params);
        offset += tangential_params.nrows();
        result
            .fixed_slice_mut::<P::NumParameters, U1>(offset, 0)
            .copy_from(&prism_params_x);
        offset += prism_params_x.nrows();
        result
            .fixed_slice_mut::<P::NumParameters, U1>(offset, 0)
            .copy_from(&prism_params_y);
        offset += prism_params_y.nrows();
        result
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
        let point = self.linear.calibrate(point).0;
        let point = self.unproject(point);
        let point = self.undistort(point);
        NormalizedKeyPoint(point)
    }

    fn uncalibrate(&self, projection: Self::Projection) -> KeyPoint {
        let point = self.distort(projection.0);
        let point = self.project(point);
        self.linear.uncalibrate(NormalizedKeyPoint(point))
    }
}

pub mod models {
    use super::Camera;
    use crate::distortion_function::{self, Constant, DistortionFunction, Polynomial, Rational};
    use crate::CameraIntrinsics;
    use cv_core::nalgebra::{Matrix3, Vector3, VectorN, U1, U8};
    use cv_core::nalgebra::{U3, U4};

    /// Generate Adobe rectilinear camera model
    ///
    /// $u_0, v_0, f_x, f_y, k_1, k_2, k_3, k_4, k_5$
    ///
    /// # References
    ///
    /// http://download.macromedia.com/pub/labs/lensprofile_creator/lensprofile_creator_cameramodel.pdf
    ///
    /// # To do
    ///
    /// * Implement Geometric Distortion Model for Fisheye Lenses
    /// * Implement Lateral Chromatic Aberration Model
    /// * Implement Vignette Model
    /// * Read/Write Lens Correction Profile File Format
    pub type Adobe = Camera<Polynomial<U3>, Constant, Constant>;

    impl Adobe {
        pub fn adobe(_parameters: [f64; 9]) -> Adobe {
            let mut camera = Adobe::default();
            todo!()
        }
    }

    /// OpenCV compatible model with up to 12 distortion coefficients
    pub type OpenCV = Camera<Rational<U4, U4>, Constant, Polynomial<U3>>;

    impl OpenCV {
        /// Generate OpenCV camera with up to twelve distortion coefficients.
        pub fn opencv(camera_matrix: Matrix3<f64>, distortion_coefficients: &[f64]) -> OpenCV {
            // Zero pad coefficients
            assert!(
                distortion_coefficients.len() <= 12,
                "Up to 12 coefficients are supported."
            );
            let mut coeffs = [0.0; 12];
            coeffs.copy_from_slice(distortion_coefficients);

            // Construct components
            let mut camera = OpenCV::default();
            camera.linear = CameraIntrinsics::from_matrix(camera_matrix);
            camera.radial_distortion =
                Rational::from_parameters(VectorN::from_column_slice_generic(
                    U8,
                    U1,
                    &[
                        1.0, coeffs[0], coeffs[1], coeffs[4], 1.0, coeffs[5], coeffs[6], coeffs[7],
                    ],
                ));
            camera.tangential = [coeffs[2], coeffs[3]];
            camera.tangential_distortion = distortion_function::one();
            camera.prism_distortion = [
                Polynomial::<U3>::from_parameters(Vector3::new(0.0, coeffs[8], coeffs[9])),
                Polynomial::<U3>::from_parameters(Vector3::new(0.0, coeffs[10], coeffs[11])),
            ];
            camera
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cv_core::nalgebra::{Matrix3, Point2};
    use float_eq::assert_float_eq;
    use models::OpenCV;

    #[rustfmt::skip]
    const UNDISTORTED: &[[f64; 2]] = &[
        [-2.678325867593669   , -2.018833440108032   ],
        [-0.8278750155570158  , -1.2269359155245612  ],
        [-0.02009583871305682 , -1.0754641516207084  ],
        [ 0.7696051515407775  , -1.1999092188198324  ],
        [ 2.4240231486605044  , -1.8482850962222797  ],
        [-2.00590387018105    , -0.7622775647671782  ],
        [-0.6803517193354109  , -0.508744726065251   ],
        [-0.018858004780545452, -0.45712295391460256 ],
        [ 0.629841408485172   , -0.5002154916071485  ],
        [ 1.852035193756623   , -0.7179892894252142  ],
        [-1.8327338936360238  , -0.01592369353657851 ],
        [-0.6414183824611562  , -0.01250548415320514 ],
        [-0.01831155521928233 , -0.011537853508065351],
        [ 0.5932054070590098  , -0.012346500967552635],
        [ 1.6999857018318918  , -0.015398552060458103],
        [-2.0026791707814664  ,  0.7276895662410346  ],
        [-0.6772278897320149  ,  0.48030562044200287 ],
        [-0.01882077458202271 ,  0.4309830892825161  ],
        [ 0.6268195195673317  ,  0.4721702706067835  ],
        [ 1.8474518498275845  ,  0.6841838156568198  ],
        [-2.7191252091392095  ,  2.00821485753557    ],
        [-0.8213459272712429  ,  1.187222601515949   ],
        [-0.020100153303183495,  1.0384037280790197  ],
        [ 0.762958127238301   ,  1.1607439280019045  ],
        [ 2.446040772535413   ,  1.8275603904250888  ],
    ];

    #[rustfmt::skip]
    const DISTORTED: &[[f64; 2]] = &[
        [-1.1422163009074084  , -0.8548601705472734  ],
        [-0.5802629659506781  , -0.8548601705472634  ],
        [-0.018309630993960678, -0.8548601705472749  ],
        [ 0.5436437039627523  , -0.8548601705472572  ],
        [ 1.105597038919486   , -0.8548601705472729  ],
        [-1.1422163009074044  , -0.4331982294741029  ],
        [-0.5802629659506768  , -0.4331982294740987  ],
        [-0.018309630993960376, -0.43319822947409947 ],
        [ 0.5436437039627607  , -0.4331982294741025  ],
        [ 1.1055970389194907  , -0.4331982294741061  ],
        [-1.1422163009074042  , -0.011536288400935572],
        [-0.58026296595067    , -0.011536288400935261],
        [-0.018309630993960747, -0.011536288400935736],
        [ 0.5436437039627555  , -0.011536288400935357],
        [ 1.1055970389194847  , -0.011536288400935575],
        [-1.1422163009074064  ,  0.4101256526722325  ],
        [-0.5802629659506844  ,  0.4101256526722334  ],
        [-0.018309630993960612,  0.41012565267223955 ],
        [ 0.5436437039627673  ,  0.4101256526722365  ],
        [ 1.1055970389194787  ,  0.41012565267223033 ],
        [-1.1422163009074024  ,  0.8317875937453983  ],
        [-0.5802629659506786  ,  0.8317875937453932  ],
        [-0.018309630993960567,  0.8317875937453815  ],
        [ 0.5436437039627544  ,  0.8317875937453895  ],
        [ 1.1055970389194876  ,  0.8317875937454029  ],
    ];

    fn camera_1() -> OpenCV {
        // Calibration parameters for a GoPro Hero 6 using OpenCV
        #[rustfmt::skip]
        let camera_matrix = Matrix3::from_row_slice(&[
            1.77950719e+03, 0.00000000e+00, 2.03258212e+03,
            0.00000000e+00, 1.77867606e+03, 1.52051932e+03,
            0.00000000e+00, 0.00000000e+00, 1.00000000e+00]);
        #[rustfmt::skip]
        let distortion: [f64; 12] = [
            2.56740016e+01,  1.52496764e+01, -5.01712057e-04,
            1.09310463e-03,  6.72953083e-01,  2.59544797e+01,
            2.24213453e+01,  3.04318306e+00, -3.23278793e-03,
            9.53176056e-05, -9.35687185e-05,  2.96341863e-05];
        OpenCV::opencv(camera_matrix, &distortion)
    }

    #[test]
    fn test_distort_1() {
        let camera = camera_1();
        for (undistorted, expected) in UNDISTORTED.iter().zip(DISTORTED.iter()) {
            let [x, y] = *undistorted;
            let [ex, ey] = *expected;

            let distorted = camera.distort(Point2::new(x, y));
            let (x, y) = (distorted[0], distorted[1]);

            assert_float_eq!(x, ex, ulps <= 2);
            assert_float_eq!(y, ey, ulps <= 2);
        }
    }

    #[test]
    fn test_undistort_1() {
        let camera = camera_1();
        for (expected, distorted) in UNDISTORTED.iter().zip(DISTORTED.iter()) {
            let [x, y] = *distorted;
            let [ex, ey] = *expected;

            let distorted = Point2::new(x, y);
            let undistorted = camera.undistort(distorted);
            let (x, y) = (undistorted[0], undistorted[1]);

            assert_float_eq!(x, ex, ulps <= 7);
            assert_float_eq!(y, ey, ulps <= 7);
        }
    }
}
