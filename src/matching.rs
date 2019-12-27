use derive_more::Constructor;
use nalgebra::Point2;

/// Normalized keypoint match
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Constructor)]
pub struct KeypointMatch(Point2<f32>, Point2<f32>);
