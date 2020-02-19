use derive_more::{AsMut, AsRef, Deref, DerefMut, From, Into};
use nalgebra::{Matrix3, Point2, Point3, Vector2, Vector3};

/// A point on an image frame. This type should only be used when
/// the point location is on the image frame in pixel coordinates.
/// This means the keypoint is neither undistorted nor normalized.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct ImageKeyPoint(pub Point2<f64>);

/// A 3d point which is relative to the camera's optical center and orientation where
/// the positive Y axis is up and positive Z axis is forwards from the center of the
/// camera. The unit of distance of a `CameraPoint` is unspecified and relative to
/// the current reconstruction.
///
/// A `CameraPoint` can be turned into a [`NormalizedKeyPoint`] by using the `Into` or
/// `From` impl. This is done by projecting the `CameraPoint` onto the virtual plane
/// at a depth `z = 1.0`. The operation cannot be done in reverse because the depth
/// (`z` component) or distance from optical center (length) is unknown.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct CameraPoint(pub Point3<f64>);

/// A point in normalized image coordinates. This keypoint has been corrected
/// for distortion and normalized based on the camrea intrinsic matrix.
/// Please note that the intrinsic matrix accounts for the natural focal length
/// and any magnification to the image. Ultimately, the key points must be
/// represented by their position on the camera sensor and normalized to the
/// focal length of the camera.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct NormalizedKeyPoint(pub Point2<f64>);

impl NormalizedKeyPoint {
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
        CameraPoint((self.coords * depth).push(depth).into())
    }

    /// Projects the keypoint out to the [`CameraPoint`] that is
    /// `distance` away from the optical center of the camera. This
    /// `distance` is defined as the norm of the vector that represents
    /// the position delta of the point from the camera.
    pub fn with_distance(self, distance: f64) -> CameraPoint {
        CameraPoint((distance * self.bearing()).into())
    }

    /// Get the epipolar point as a [`CameraPoint`].
    ///
    /// The epipolar point is the point that is formed on the virtual
    /// image at a depth 1.0 in front of the camera. For that reason,
    /// this is the exact same as calling `nkp.with_depth(1.0)`.
    pub fn epipolar_point(self) -> CameraPoint {
        self.with_depth(1.0)
    }

    /// Returns a unit vector of the direction that the epipolar line
    /// created by this `NormalizedKeyPoint` projects out of the
    /// optical center of the camera. This is defined as the the
    /// normalized position delta of the epipolar point from the
    /// optical center of the camera.
    pub fn bearing(self) -> Vector3<f64> {
        self.0.coords.push(1.0).normalize()
    }

    /// Same as [`bearing`], but it is returned unnormalized.
    pub fn bearing_unnormalized(self) -> Vector3<f64> {
        self.0.coords.push(1.0).normalize()
    }
}

impl From<CameraPoint> for NormalizedKeyPoint {
    fn from(CameraPoint(point): CameraPoint) -> Self {
        NormalizedKeyPoint(point.xy() / point.z)
    }
}

/// This contains intrinsic camera parameters as per
/// [this Wikipedia page](https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters).
///
/// For a high quality camera, this may be sufficient to normalize image coordinates.
/// Undistortion may also be necessary to normalize image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
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

    /// Takes in an [`ImageKeyPoint`] from an image in pixel coordinates and
    /// converts it to a [`NormalizedKeyPoint`].
    ///
    /// ```
    /// # use cv_core::{ImageKeyPoint, NormalizedKeyPoint, CameraIntrinsics};
    /// # use cv_core::nalgebra::{Vector2, Vector3, Point2};
    /// let intrinsics = CameraIntrinsics {
    ///     focals: Vector2::new(800.0, 900.0),
    ///     principal_point: Point2::new(500.0, 600.0),
    ///     skew: 1.7,
    /// };
    /// let kp = ImageKeyPoint(Point2::new(471.0, 322.0));
    /// let nkp = intrinsics.normalize(kp);
    /// let calibration_matrix = intrinsics.matrix();
    /// let distance = (kp.to_homogeneous() - calibration_matrix * nkp.to_homogeneous()).norm();
    /// assert!(distance < 0.1);
    /// ```
    pub fn normalize(&self, image: ImageKeyPoint) -> NormalizedKeyPoint {
        let ImageKeyPoint(image) = image;
        let centered = image - self.principal_point;
        let y = centered.y / self.focals.y;
        let x = (centered.x - self.skew * y) / self.focals.x;
        NormalizedKeyPoint(Point2::new(x, y))
    }
}

/// This contains basic camera specifications that one could find on a
/// manufacturer's website. This only contains parameters that cannot
/// be changed about a camera. The focal length is not included since
/// that can typically be changed and images can also be magnified.
///
/// All distance units should be in meters to avoid conversion issues.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
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
