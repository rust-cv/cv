use crate::KeyPointWorldMatch;
use derive_more::{AsMut, AsRef, Constructor, Deref, DerefMut, From, Into};
use nalgebra::{Isometry3};
use sample_consensus::Model;

/// This contains a world pose, which is a pose of the world relative to the camera.
/// This transforms world points into camera points. These camera points are 3d
/// and the `z` axis represents the depth. Projecting these points onto the plane
/// at `z = 1` will tell you where the points are in normalized image coordinates on the image.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, Constructor, Deref, DerefMut, From, Into)]
pub struct WorldPose(pub Isometry3<f32>);

impl Model<KeyPointWorldMatch> for WorldPose {
    fn residual(&self, data: &KeyPointWorldMatch) -> f32 {
        let WorldPose(iso) = *self;
        let KeyPointWorldMatch(camera, world) = *data;

        let new_bearing = (iso * world.coords).normalize();
        let bearing_vector = camera.to_homogeneous().normalize();
        1.0 - bearing_vector.dot(&new_bearing)
    }
}

impl From<CameraPose> for WorldPose {
    fn from(camera: CameraPose) -> Self {
        Self(camera.inverse())
    }
}

/// This contains a camera pose, which is a pose of the camera relative to the world.
/// This transforms camera points (with depth as `z`) into world coordinates.
/// This also tells you where the camera is located and oriented in the world.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, Constructor, Deref, DerefMut, From, Into)]
pub struct CameraPose(pub Isometry3<f32>);

impl From<WorldPose> for CameraPose {
    fn from(world: WorldPose) -> Self {
        Self(world.inverse())
    }
}
