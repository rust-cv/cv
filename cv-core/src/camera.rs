use crate::{ImagePoint, KeyPoint};
use nalgebra::{Unit, Vector3};

/// Describes the direction that the projection onto the camera's optical center
/// came from. It is implemented on projection items from different camera models.
/// It is also implemented for `Unit<Vector3<f64>>` if you want to pre-compute the
/// normalized bearings for efficiency or to turn all camera models into a unified type.
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

    /// Converts a bearing vector back into this bearing type.
    ///
    /// This is useful if you would like to go backwards from reconstruction space to image space.
    /// See [`CameraModel::uncalibrate`] for how to then convert the camera bearing into image coordinates.
    fn from_bearing_vector(bearing: Vector3<f64>) -> Self;

    /// Converts a bearing unit vector back into this bearing type.
    ///
    /// This is useful if you would like to go backwards from reconstruction space to image space.
    /// See [`CameraModel::uncalibrate`] for how to then convert the camera bearing into image coordinates.
    fn from_bearing_unit_vector(bearing: Unit<Vector3<f64>>) -> Self
    where
        Self: Sized,
    {
        Self::from_bearing_vector(bearing.into_inner())
    }
}

impl Bearing for Unit<Vector3<f64>> {
    fn bearing(&self) -> Unit<Vector3<f64>> {
        *self
    }

    fn bearing_unnormalized(&self) -> Vector3<f64> {
        self.into_inner()
    }

    fn from_bearing_vector(bearing: Vector3<f64>) -> Self {
        Unit::new_normalize(bearing)
    }

    #[allow(clippy::wrong_self_convention)]
    fn from_bearing_unit_vector(bearing: Unit<Vector3<f64>>) -> Self {
        bearing
    }
}

/// Allows conversion between the point on an image and the internal projection
/// which can describe the bearing of the projection out of the camera.
pub trait CameraModel {
    type Projection: Bearing;

    /// Extracts a projection from a pixel location in an image.
    fn calibrate<P>(&self, point: P) -> Self::Projection
    where
        P: ImagePoint;

    /// Extracts the pixel location in the image from the projection.
    fn uncalibrate(&self, projection: Self::Projection) -> KeyPoint;
}
