use derive_more::{AsMut, AsRef, Deref, DerefMut, From, Into};
use nalgebra::Point2;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

/// Allows the retrieval of the point on the image the feature came from.
pub trait ImagePoint {
    /// Retrieves the point on the image
    fn image_point(&self) -> Point2<f64>;
}

/// A point on an image frame. This type should be used when
/// the point location is on the image frame in pixel coordinates.
/// This means the keypoint is neither undistorted nor normalized.
///
/// For calibrated coordinates, use a type that implements [`Bearing`](crate::Bearing).
/// This can be a type from a camera model crate (like `cv-pinhole`), or
/// it can be the `Unit<Vector3<f64>>` type, which implements bearing.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsMut, AsRef, Deref, DerefMut, From, Into)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct KeyPoint(pub Point2<f64>);

impl ImagePoint for KeyPoint {
    fn image_point(&self) -> Point2<f64> {
        self.0
    }
}
