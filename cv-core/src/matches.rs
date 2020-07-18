use crate::WorldPoint;

/// Normalized keypoint match
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FeatureMatch<P>(pub P, pub P);

/// Normalized keypoint to world point match
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct FeatureWorldMatch<P>(pub P, pub WorldPoint);
