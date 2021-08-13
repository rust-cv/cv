use crate::{ImagePoint, KeyPoint};
use nalgebra::UnitVector3;

/// Allows conversion between the point on an image and the internal projection
/// which can describe the bearing of the projection out of the camera.
pub trait CameraModel {
    /// Extracts a bearing from a pixel location in an image.
    ///
    /// Note that while the image point's Y axis points downwards (origin in top left),
    /// the bearing's Y axis points upwards (origin in bottom left).
    ///
    /// The bearings X axis points right, Y axis points up, and Z axis points forwards.
    fn calibrate<P>(&self, point: P) -> UnitVector3<f64>
    where
        P: ImagePoint;

    /// Extracts the pixel location in the image from the bearing.
    ///
    /// Note that while the image point's Y axis points downwards (origin in top left),
    /// the bearing's Y axis points upwards (origin in bottom left).
    ///
    /// The bearings X axis points right, Y axis points up, and Z axis points forwards.
    ///
    /// Since this might not be possible (if bearing is behind camera for pinhole camera),
    /// this operation is fallible.
    fn uncalibrate(&self, bearing: UnitVector3<f64>) -> Option<KeyPoint>;
}
