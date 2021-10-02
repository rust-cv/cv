use derive_more::{AsMut, AsRef, Deref, DerefMut, From, Into};
use nalgebra::Point2;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

/// Allows the retrieval of the point on the image the feature came from.
///
/// The origin for an image point is in the top left of the image. Positive X axis points right
/// and positive Y axis points down.
pub trait ImagePoint {
    /// Retrieves the point on the image
    fn image_point(&self) -> Point2<f64>;
}

/// A point on an image frame. This type should be used when
/// the point location is on the image frame in pixel coordinates.
/// This means the keypoint is neither undistorted nor normalized.
///
/// For calibrated coordinates, you need to use an appropriate camera model crate (like `cv-pinhole`).
/// These crates convert image coordinates into bearings. For more information, see the trait definition
/// [`crate::CameraModel`].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsMut, AsRef, Deref, DerefMut, From, Into)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct KeyPoint(pub Point2<f64>);

impl ImagePoint for KeyPoint {
    fn image_point(&self) -> Point2<f64> {
        self.0
    }
}
