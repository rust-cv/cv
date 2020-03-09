use cv_core::nalgebra::Point2;
use cv_core::ImagePoint;

/// A point of interest in an image.
/// This pretty much follows from OpenCV conventions.
#[derive(Debug, Clone, Copy)]
pub struct Keypoint {
    /// The horizontal coordinate in a coordinate system is
    /// defined s.t. +x faces right and starts from the top
    /// of the image.
    /// the vertical coordinate in a coordinate system is defined
    /// s.t. +y faces toward the bottom of an image and starts
    /// from the left side of the image.
    pub point: (f32, f32),
    /// The magnitude of response from the detector.
    pub response: f32,

    /// The radius defining the extent of the keypoint, in pixel units
    pub size: f32,

    /// The level of scale space in which the keypoint was detected.
    pub octave: usize,

    /// A classification ID
    pub class_id: usize,

    /// The orientation angle
    pub angle: f32,
}

impl ImagePoint for Keypoint {
    fn image_point(&self) -> Point2<f64> {
        Point2::new(self.point.0 as f64, self.point.1 as f64)
    }
}

/// A feature descriptor.
#[derive(Debug, Clone)]
pub struct Descriptor {
    pub vector: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct Results {
    pub keypoints: Vec<Keypoint>,
    pub descriptors: Vec<Descriptor>,
}
