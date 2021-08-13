use crate::WorldPoint;
use nalgebra::UnitVector3;

/// Two keypoint bearings matched together from two separate images
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureMatch(pub UnitVector3<f64>, pub UnitVector3<f64>);

/// A keypoint bearing matched to a [`WorldPoint`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureWorldMatch(pub UnitVector3<f64>, pub WorldPoint);
