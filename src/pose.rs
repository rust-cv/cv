use derive_more::{AsMut, AsRef, Constructor, Deref, DerefMut, From, Into};
use nalgebra::Isometry3;

/// This contains a camera pose. The pose is in "world" coordinates.
/// This means that the real-world units of the pose are unknown, but the
/// unit of distance and orientation are the same as the current reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, Constructor, Deref, DerefMut, From, Into)]
pub struct WorldPose(Isometry3<f32>);
