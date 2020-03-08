#[cfg(feature = "pinhole")]
pub mod pinhole;

use crate::{ImageKeyPoint, ImagePoint};
use derive_more::{AsMut, AsRef, Deref, DerefMut, From, Into};
use nalgebra::{Point3, Unit, Vector3};

/// Describes the direction that the projection onto the camera's optical center
/// came from. It is implemented on projection items from different camera models.
pub trait Bearing {
    /// Returns a unit vector of the direction that the projection
    /// created by the feature projects out of the
    /// optical center of the camera. This is defined as the the
    /// position delta of the feature from the optical center of the camera.
    fn bearing(&self) -> Unit<Vector3<f64>> {
        Unit::new_normalize(self.bearing_unnormalized())
    }

    /// Returns the unnormalized bearing which has direction point towards the direction
    /// that the signal entered the camera's center. The magnitude of this vector
    /// is unknown. Use this if you are sure that you do not need a normalized
    /// bearing. This may be faster.
    fn bearing_unnormalized(&self) -> Vector3<f64>;
}

/// A 3d point which is relative to the camera's optical center and orientation where
/// the positive X axis is right, positive Y axis is down, and positive Z axis is forwards
/// from the optical center of the camera. The unit of distance of a `CameraPoint` is
/// unspecified and relative to the current reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct CameraPoint(pub Point3<f64>);

/// Allows conversion between the point on an image and the internal projection
/// which can describe the bearing of the projection out of the camera.
pub trait CameraModel {
    type Projection: Bearing;

    /// Extracts a projection from a pixel location in an image.
    fn calibrate<P>(&self, point: P) -> Self::Projection
    where
        P: ImagePoint;

    /// Extracts the pixel location in the image from the projection.
    fn uncalibrate(&self, projection: Self::Projection) -> ImageKeyPoint;
}
