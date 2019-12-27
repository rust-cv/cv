use derive_more::Constructor;
use crate::{NormalizedKeyPoint, WorldPoint};

/// Normalized keypoint match
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Constructor)]
pub struct KeypointsMatch(pub NormalizedKeyPoint, pub NormalizedKeyPoint);

/// Normalized keypoint to world point match
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Constructor)]
pub struct KeypointWorldMatch(pub NormalizedKeyPoint, pub WorldPoint);
