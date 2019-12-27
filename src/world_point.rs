use derive_more::{AsMut, AsRef, Constructor, Deref, DerefMut, From, Into};
use nalgebra::Point3;

/// A point in "world" coordinates.
/// This means that the real-world units of the pose are unknown, but the
/// unit of distance and orientation are the same as the current reconstruction.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    PartialOrd,
    AsMut,
    AsRef,
    Constructor,
    Deref,
    DerefMut,
    From,
    Into,
)]
pub struct WorldPoint(Point3<f32>);