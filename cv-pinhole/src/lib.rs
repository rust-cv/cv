//! This crate seamlessly plugs into `cv-core` and provides pinhole camera models with and without distortion correction.
//! It can be used to convert image coordinates into real 3d direction vectors (called bearings) pointing towards where
//! the light came from that hit that pixel. It can also be used to convert backwards from the 3d back to the 2d
//! using the `uncalibrate` method from the [`cv_core::CameraModel`] trait.

#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

mod essential;

pub use essential::*;

use cv_core::nalgebra::{Matrix3, Point2, Point3, Vector2, Vector3};
use cv_core::{
    Bearing, CameraModel, CameraPoint, CameraToCamera, FeatureMatch, ImagePoint, KeyPoint, Pose,
    Projective, TriangulatorRelative,
};
use derive_more::{AsMut, AsRef, Deref, DerefMut, From, Into};
use num_traits::Float;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

/// A point in normalized image coordinates. This keypoint has been corrected
/// for distortion and normalized based on the camrea intrinsic matrix.
/// Please note that the intrinsic matrix accounts for the natural focal length
/// and any magnification to the image. Ultimately, the key points must be
/// represented by their position on the camera sensor and normalized to the
/// focal length of the camera.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsMut, AsRef, Deref, DerefMut, From, Into)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct NormalizedKeyPoint(pub Point2<f64>);

impl NormalizedKeyPoint {
    /// Tries to convert the [`CameraPoint`] into a [`NormalizedKeyPoint`], but it may fail
    /// in extreme conditions, in which case `None` is returned.
    pub fn from_camera_point(point: CameraPoint) -> Option<Self> {
        Point2::from_homogeneous(point.bearing_unnormalized()).map(Self)
    }

    /// Conceptually appends a `1.0` component to the normalized keypoint to create
    /// a [`CameraPoint`] on the virtual image plane and then multiplies
    /// the point by `depth`. This `z`/`depth` component must be the depth of
    /// the keypoint in the direction the camera is pointing from the
    /// camera's optical center.
    ///
    /// The `depth` is computed as the dot product of the unit camera norm
    /// with the vector that represents the position delta of the point from
    /// the camera.
    pub fn with_depth(self, depth: f64) -> CameraPoint {
        (self.coords * depth).push(depth).to_homogeneous().into()
    }

    /// Projects the keypoint out to the [`CameraPoint`] that is
    /// `distance` away from the optical center of the camera. This
    /// `distance` is defined as the norm of the vector that represents
    /// the position delta of the point from the camera.
    pub fn with_distance(self, distance: f64) -> CameraPoint {
        (distance * *self.bearing()).to_homogeneous().into()
    }

    /// Get the virtual image point as a [`Point3`].
    ///
    /// The virtual image point is the point that is formed on the virtual
    /// image plane at a depth 1.0 in front of the camera.
    pub fn virtual_image_point(self) -> Point3<f64> {
        self.coords.push(1.0).into()
    }
}

impl Bearing for NormalizedKeyPoint {
    fn bearing_unnormalized(&self) -> Vector3<f64> {
        self.0.coords.push(1.0)
    }

    fn from_bearing_vector(bearing: Vector3<f64>) -> Self {
        Self((bearing.xy() / bearing.z).into())
    }
}

/// This contains intrinsic camera parameters as per
/// [this Wikipedia page](https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters).
///
/// For a high quality camera, this may be sufficient to normalize image coordinates.
/// Undistortion may also be necessary to normalize image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct CameraIntrinsics {
    pub focals: Vector2<f64>,
    pub principal_point: Point2<f64>,
    pub skew: f64,
}

impl CameraIntrinsics {
    /// Creates camera intrinsics that would create an identity intrinsic matrix.
    /// This would imply that the pixel positions have an origin at `0,0`,
    /// the pixel distance unit is the focal length, pixels are square,
    /// and there is no skew.
    pub fn identity() -> Self {
        Self {
            focals: Vector2::new(1.0, 1.0),
            skew: 0.0,
            principal_point: Point2::new(0.0, 0.0),
        }
    }

    pub fn focals(self, focals: Vector2<f64>) -> Self {
        Self { focals, ..self }
    }

    pub fn focal(self, focal: f64) -> Self {
        Self {
            focals: Vector2::new(focal, focal),
            ..self
        }
    }

    pub fn principal_point(self, principal_point: Point2<f64>) -> Self {
        Self {
            principal_point,
            ..self
        }
    }

    pub fn skew(self, skew: f64) -> Self {
        Self { skew, ..self }
    }

    #[rustfmt::skip]
    pub fn matrix(&self) -> Matrix3<f64> {
        Matrix3::new(
            self.focals.x,  self.skew,      self.principal_point.x,
            0.0,            self.focals.y,  self.principal_point.y,
            0.0,            0.0,            1.0,
        )
    }
}

impl CameraModel for CameraIntrinsics {
    type Projection = NormalizedKeyPoint;

    /// Takes in a point from an image in pixel coordinates and
    /// converts it to a [`NormalizedKeyPoint`].
    ///
    /// ```
    /// use cv_core::{KeyPoint, CameraModel};
    /// use cv_pinhole::{NormalizedKeyPoint, CameraIntrinsics};
    /// use cv_core::nalgebra::{Vector2, Vector3, Point2};
    /// let intrinsics = CameraIntrinsics {
    ///     focals: Vector2::new(800.0, 900.0),
    ///     principal_point: Point2::new(500.0, 600.0),
    ///     skew: 1.7,
    /// };
    /// let kp = KeyPoint(Point2::new(471.0, 322.0));
    /// let nkp = intrinsics.calibrate(kp);
    /// let calibration_matrix = intrinsics.matrix();
    /// let distance = (kp.to_homogeneous() - calibration_matrix * nkp.to_homogeneous()).norm();
    /// assert!(distance < 0.1);
    /// ```
    fn calibrate<P>(&self, point: P) -> NormalizedKeyPoint
    where
        P: ImagePoint,
    {
        let centered = point.image_point() - self.principal_point;
        let y = centered.y / self.focals.y;
        let x = (centered.x - self.skew * y) / self.focals.x;
        NormalizedKeyPoint(Point2::new(x, y))
    }

    /// Converts a [`NormalizedKeyPoint`] back into pixel coordinates.
    ///
    /// ```
    /// use cv_core::{KeyPoint, CameraModel};
    /// use cv_pinhole::{NormalizedKeyPoint, CameraIntrinsics};
    /// use cv_core::nalgebra::{Vector2, Vector3, Point2};
    /// let intrinsics = CameraIntrinsics {
    ///     focals: Vector2::new(800.0, 900.0),
    ///     principal_point: Point2::new(500.0, 600.0),
    ///     skew: 1.7,
    /// };
    /// let kp = KeyPoint(Point2::new(471.0, 322.0));
    /// let nkp = intrinsics.calibrate(kp);
    /// let ukp = intrinsics.uncalibrate(nkp);
    /// assert!((kp.0 - ukp.0).norm() < 1e-6);
    /// ```
    fn uncalibrate(&self, projection: NormalizedKeyPoint) -> KeyPoint {
        let y = projection.y * self.focals.y;
        let x = projection.x * self.focals.x + self.skew * projection.y;
        let centered = Point2::new(x, y);
        KeyPoint(centered + self.principal_point.coords)
    }
}

/// This contains intrinsic camera parameters as per
/// [this Wikipedia page](https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters).
///
/// This also performs undistortion by applying one radial distortion coefficient (K1).
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct CameraIntrinsicsK1Distortion {
    pub simple_intrinsics: CameraIntrinsics,
    pub k1: f64,
}

impl CameraIntrinsicsK1Distortion {
    /// Creates the camera intrinsics using simple intrinsics with no distortion and a K1 distortion coefficient.
    pub fn new(simple_intrinsics: CameraIntrinsics, k1: f64) -> Self {
        Self {
            simple_intrinsics,
            k1,
        }
    }
}

impl CameraModel for CameraIntrinsicsK1Distortion {
    type Projection = NormalizedKeyPoint;

    /// Takes in a point from an image in pixel coordinates and
    /// converts it to a [`NormalizedKeyPoint`].
    ///
    /// ```
    /// use cv_core::{KeyPoint, CameraModel};
    /// use cv_pinhole::{NormalizedKeyPoint, CameraIntrinsics, CameraIntrinsicsK1Distortion};
    /// use cv_core::nalgebra::{Vector2, Vector3, Point2};
    /// let intrinsics = CameraIntrinsics {
    ///     focals: Vector2::new(800.0, 900.0),
    ///     principal_point: Point2::new(500.0, 600.0),
    ///     skew: 1.7,
    /// };
    /// let k1 = -0.164624;
    /// let intrinsics = CameraIntrinsicsK1Distortion::new(
    ///     intrinsics,
    ///     k1,
    /// );
    /// let kp = KeyPoint(Point2::new(471.0, 322.0));
    /// let nkp = intrinsics.calibrate(kp);
    /// let simple_nkp = intrinsics.simple_intrinsics.calibrate(kp);
    /// let distance = (nkp.0.coords - (simple_nkp.0.coords / (1.0 + k1 * simple_nkp.0.coords.norm_squared()))).norm();
    /// assert!(distance < 0.1);
    /// ```
    fn calibrate<P>(&self, point: P) -> NormalizedKeyPoint
    where
        P: ImagePoint,
    {
        let NormalizedKeyPoint(distorted) = self.simple_intrinsics.calibrate(point);
        let r2 = distorted.coords.norm_squared();
        let undistorted = (distorted.coords / (1.0 + self.k1 * r2)).into();

        NormalizedKeyPoint(undistorted)
    }

    /// Converts a [`NormalizedKeyPoint`] back into pixel coordinates.
    ///
    /// ```
    /// use cv_core::{KeyPoint, CameraModel};
    /// use cv_pinhole::{NormalizedKeyPoint, CameraIntrinsics, CameraIntrinsicsK1Distortion};
    /// use cv_core::nalgebra::{Vector2, Vector3, Point2};
    /// let intrinsics = CameraIntrinsics {
    ///     focals: Vector2::new(800.0, 900.0),
    ///     principal_point: Point2::new(500.0, 600.0),
    ///     skew: 1.7,
    /// };
    /// let intrinsics = CameraIntrinsicsK1Distortion::new(
    ///     intrinsics,
    ///     -0.164624,
    /// );
    /// let kp = KeyPoint(Point2::new(471.0, 322.0));
    /// let nkp = intrinsics.calibrate(kp);
    /// let ukp = intrinsics.uncalibrate(nkp);
    /// assert!((kp.0 - ukp.0).norm() < 1e-6, "{:?}", (kp.0 - ukp.0).norm());
    /// ```
    fn uncalibrate(&self, projection: NormalizedKeyPoint) -> KeyPoint {
        let NormalizedKeyPoint(undistorted) = projection;
        // This was not easy to compute, but you can set up a quadratic to solve
        // for r^2 with the undistorted keypoint. This is the result.
        let u2 = undistorted.coords.norm_squared();
        // This is actually r^2 * k1.
        let r2_mul_k1 = -(2.0 * self.k1 * u2 + Float::sqrt(1.0 - 4.0 * self.k1 * u2) - 1.0)
            / (2.0 * self.k1 * u2);
        self.simple_intrinsics.uncalibrate(NormalizedKeyPoint(
            (undistorted.coords * (1.0 + r2_mul_k1)).into(),
        ))
    }
}

/// This contains basic camera specifications that one could find on a
/// manufacturer's website. This only contains parameters that cannot
/// be changed about a camera. The focal length is not included since
/// that can typically be changed and images can also be magnified.
///
/// All distance units should be in meters to avoid conversion issues.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct CameraSpecification {
    pub pixels: Vector2<usize>,
    pub pixel_dimensions: Vector2<f64>,
}

impl CameraSpecification {
    /// Creates a [`CameraSpecification`] using the sensor dimensions.
    pub fn from_sensor(pixels: Vector2<usize>, sensor_dimensions: Vector2<f64>) -> Self {
        Self {
            pixels,
            pixel_dimensions: Vector2::new(
                sensor_dimensions.x / pixels.x as f64,
                sensor_dimensions.y / pixels.y as f64,
            ),
        }
    }

    /// Creates a [`CameraSpecification`] using the sensor width assuming a square pixel.
    pub fn from_sensor_square(pixels: Vector2<usize>, sensor_width: f64) -> Self {
        let pixel_width = sensor_width / pixels.x as f64;
        Self {
            pixels,
            pixel_dimensions: Vector2::new(pixel_width, pixel_width),
        }
    }

    /// Combines the [`CameraSpecification`] with a focal length to create a [`CameraIntrinsics`].
    ///
    /// This assumes square pixels and a perfectly centered principal point.
    pub fn intrinsics_centered(&self, focal: f64) -> CameraIntrinsics {
        CameraIntrinsics::identity()
            .focal(focal)
            .principal_point(self.pixel_dimensions.map(|p| p as f64 / 2.0 - 0.5).into())
    }
}

/// Find the reprojection error in focal lengths of a feature match and a relative pose using the given triangulator.
///
/// If the feature match destructures as `FeatureMatch(a, b)`, then A is the camera of `a`, and B is the camera of `b`.
/// The pose must transform the space of camera A into camera B. The triangulator will triangulate the 3d point from the
/// perspective of camera A, and the pose will be used to transform the point into the perspective of camera B.
///
/// ```
/// use cv_core::{CameraToCamera, CameraPoint, FeatureMatch, Pose};
/// use cv_core::nalgebra::{Point3, IsometryMatrix3, Vector3, Rotation3};
/// use cv_pinhole::NormalizedKeyPoint;
/// // Create an arbitrary point in the space of camera A.
/// let point_a = CameraPoint(Point3::new(0.4, -0.25, 5.0).to_homogeneous());
/// // Create an arbitrary relative pose between two cameras A and B.
/// let pose = CameraToCamera::from_parts(Vector3::new(0.1, 0.2, -0.5), Rotation3::identity());
/// // Transform the point in camera A to camera B.
/// let point_b = pose.transform(point_a);
///
/// // Convert the camera points to normalized image coordinates.
/// let nkpa = NormalizedKeyPoint::from_camera_point(point_a).unwrap();
/// let nkpb = NormalizedKeyPoint::from_camera_point(point_b).unwrap();
///
/// // Create a triangulator.
/// let triangulator = cv_geom::MinSquaresTriangulator::new();
///
/// // Since the normalized keypoints were computed exactly, there should be no reprojection error.
/// let errors = cv_pinhole::pose_reprojection_error(pose, FeatureMatch(nkpa, nkpb), triangulator).unwrap();
/// let average_error = errors.iter().map(|v| v.norm()).sum::<f64>() * 0.5;
/// assert!(average_error < 1e-6);
/// ```
pub fn pose_reprojection_error(
    pose: CameraToCamera,
    m: FeatureMatch<NormalizedKeyPoint>,
    triangulator: impl TriangulatorRelative,
) -> Option<[Vector2<f64>; 2]> {
    let FeatureMatch(a, b) = m;
    triangulator
        .triangulate_relative(pose, a, b)
        .and_then(|point_a| {
            let reproject_a = NormalizedKeyPoint::from_camera_point(point_a)?;
            let point_b = pose.transform(point_a);
            let reproject_b = NormalizedKeyPoint::from_camera_point(point_b)?;
            Some([a.0 - reproject_a.0, b.0 - reproject_b.0])
        })
}

/// See [`pose_reprojection_error`].
///
/// This is a convenience function that simply finds the average reprojection error rather than all components.
///
/// ```
/// use cv_core::{CameraToCamera, CameraPoint, FeatureMatch, Pose};
/// use cv_core::nalgebra::{Point3, IsometryMatrix3, Vector3, Rotation3};
/// use cv_pinhole::NormalizedKeyPoint;
/// // Create an arbitrary point in the space of camera A.
/// let point_a = CameraPoint(Point3::new(0.4, -0.25, 5.0).to_homogeneous());
/// // Create an arbitrary relative pose between two cameras A and B.
/// let pose = CameraToCamera::from_parts(Vector3::new(0.1, 0.2, -0.5), Rotation3::identity());
/// // Transform the point in camera A to camera B.
/// let point_b = pose.transform(point_a);
///
/// // Convert the camera points to normalized image coordinates.
/// let nkpa = NormalizedKeyPoint::from_camera_point(point_a).unwrap();
/// let nkpb = NormalizedKeyPoint::from_camera_point(point_b).unwrap();
///
/// // Create a triangulator.
/// let triangulator = cv_geom::MinSquaresTriangulator::new();
///
/// // Since the normalized keypoints were computed exactly, there should be no reprojection error.
/// let average_error = cv_pinhole::average_pose_reprojection_error(pose, FeatureMatch(nkpa, nkpb), triangulator).unwrap();
/// assert!(average_error < 1e-6);
/// ```
pub fn average_pose_reprojection_error(
    pose: CameraToCamera,
    m: FeatureMatch<NormalizedKeyPoint>,
    triangulator: impl TriangulatorRelative,
) -> Option<f64> {
    pose_reprojection_error(pose, m, triangulator)
        .map(|errors| errors.iter().map(|v| v.norm()).sum::<f64>() * 0.5)
}
