use crate::KeypointWorldMatch;
use derive_more::{AsMut, AsRef, Constructor, Deref, DerefMut, From, Into};
use nalgebra::Isometry3;
use sample_consensus::Model;

/// This contains a camera pose. The pose is in "world" coordinates.
/// This means that the real-world units of the pose are unknown, but the
/// unit of distance and orientation are the same as the current reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, Constructor, Deref, DerefMut, From, Into)]
pub struct WorldPose(Isometry3<f32>);

impl Model<KeypointWorldMatch> for WorldPose {
    fn residual(&self, data: &KeypointWorldMatch) -> f32 {
        let WorldPose(iso) = *self;
        let KeypointWorldMatch(camera, world) = *data;

        let new_bearing = (iso * world.coords).normalize();
        let bearing_vector = camera.to_homogeneous().normalize();
        1.0 - bearing_vector.dot(&new_bearing)
    }
}
