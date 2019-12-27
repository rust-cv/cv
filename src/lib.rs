pub use nalgebra;

use derive_more::{AsMut, AsRef, Constructor, Deref, DerefMut, From, Into};
use nalgebra::{Matrix3, Point2, Vector2};

/// A point on an image frame. This type should only be used when
/// the point location is on the image frame in pixel coordinates.
/// This means the keypoint is neither undistorted nor normalized.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    PartialOrd,
    AsMut,
    AsRef,
    Constructor,
    Deref,
    DerefMut,
    From,
    Into,
)]
pub struct ImageKeyPoint(Point2<f32>);

/// A point in normalized image coordinates. This keypoint has been corrected
/// for distortion and normalized based on the camrea intrinsic matrix.
/// Please note that the intrinsic matrix accounts for the natural focal length
/// and any magnification to the image. Ultimately, the key points must be
/// represented by their position on the camera sensor and normalized to the
/// focal length of the camera.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    PartialOrd,
    AsMut,
    AsRef,
    Constructor,
    Deref,
    DerefMut,
    From,
    Into,
)]
pub struct NormalizedKeyPoint(Point2<f32>);

/// This contains intrinsic camera parameters as per
/// [this Wikipedia page](https://en.wikipedia.org/wiki/Camera_resectioning#Intrinsic_parameters).
///
/// For a high quality camera, this may be sufficient to normalize image coordinates.
/// Undistortion may also be necessary to normalize image coordinates.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Constructor)]
pub struct CameraIntrinsics {
    pub focals: Vector2<f32>,
    pub principal_point: Point2<f32>,
    pub skew: f32,
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

    pub fn focals(self, focals: Vector2<f32>) -> Self {
        Self { focals, ..self }
    }

    pub fn focal(self, focal: f32) -> Self {
        Self {
            focals: Vector2::new(focal, focal),
            ..self
        }
    }

    pub fn principal_point(self, principal_point: Point2<f32>) -> Self {
        Self {
            principal_point,
            ..self
        }
    }

    pub fn skew(self, skew: f32) -> Self {
        Self { skew, ..self }
    }

    #[rustfmt::skip]
    pub fn matrix(&self) -> Matrix3<f32> {
        Matrix3::new(
            self.focals.x,  self.skew,      self.principal_point.x,
            0.0,            self.focals.y,  self.principal_point.y,
            0.0,            0.0,            1.0,
        )
    }
}

/// This contains basic camera specifications that one could find on a
/// manufacturer's website. This only contains parameters that cannot
/// be changed about a camera. The focal length is not included since
/// that can typically be changed and images can also be magnified.
///
/// All distance units should be in meters to avoid conversion issues.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Constructor)]
pub struct CameraSpecification {
    pub pixels: Vector2<usize>,
    pub pixel_dimensions: Vector2<f32>,
}

impl CameraSpecification {
    /// Creates a [`CameraSpecification`] using the sensor dimensions.
    pub fn from_sensor(pixels: Vector2<usize>, sensor_dimensions: Vector2<f32>) -> Self {
        Self {
            pixels,
            pixel_dimensions: Vector2::new(
                sensor_dimensions.x / pixels.x as f32,
                sensor_dimensions.y / pixels.y as f32,
            ),
        }
    }

    /// Creates a [`CameraSpecification`] using the sensor width assuming a square pixel.
    pub fn from_sensor_square(pixels: Vector2<usize>, sensor_width: f32) -> Self {
        let pixel_width = sensor_width / pixels.x as f32;
        Self {
            pixels,
            pixel_dimensions: Vector2::new(pixel_width, pixel_width),
        }
    }

    /// Combines the [`CameraSpecification`] with a focal length to create a [`CameraIntrinsics`].
    ///
    /// This assumes square pixels and a perfectly centered principal point.
    pub fn intrinsics_centered(&self, focal: f32) -> CameraIntrinsics {
        CameraIntrinsics::identity()
            .focal(focal)
            .principal_point(self.pixel_dimensions.map(|p| p as f32 / 2.0 - 0.5).into())
    }
}
