use crate::{KeyPointWorldMatch, NormalizedKeyPoint, WorldPoint};
use derive_more::{AsMut, AsRef, Constructor, Deref, DerefMut, From, Into};
use nalgebra::{
    dimension::{U2, U7, U3},
    Isometry3, MatrixMN, Vector2,Matrix3, Matrix3x2
};
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
        let KeyPointWorldMatch(image, world) = *data;

        let new_bearing = (iso * world.coords).normalize();
        let bearing_vector = image.to_homogeneous().normalize();
        1.0 - bearing_vector.dot(&new_bearing)
    }
}

impl WorldPose {
    /// Computes difference between the image keypoint and the projected keypoint.
    pub fn projection_error(&self, correspondence: KeyPointWorldMatch) -> Vector2<f32> {
        let KeyPointWorldMatch(NormalizedKeyPoint(image), world) = correspondence;
        let NormalizedKeyPoint(projection) = self.project(world);
        image - projection
    }

    /// Projects the `WorldPoint` onto the camera as a `NormalizedKeyPoint`.
    pub fn project(&self, WorldPoint(point): WorldPoint) -> NormalizedKeyPoint {
        let WorldPose(iso) = *self;
        let camera_point = iso * point;
        NormalizedKeyPoint(camera_point.xy() / camera_point.z)
    }

    /// Computes the Jacobian of the projection in respect to the `WorldPose`.
    /// The Jacobian is in the format:
    /// ```no_build,no_run
    /// | dx/dtx dy/dPx |
    /// | dx/dty dy/dPy |
    /// | dx/dtz dy/dPz |
    /// | dx/dqr dy/dqr |
    /// | dx/dqx dy/dqx |
    /// | dx/dqy dy/dqy |
    /// | dx/dqz dy/dqz |
    /// ```
    ///
    /// Where `t` refers to the translation vector and `q` refers to the unit quaternion.
    #[rustfmt::skip]
    pub fn projection_pose_jacobian(&self, point: WorldPoint) -> MatrixMN<f32, U7, U2> {
        let q = self.0.rotation.quaternion().coords;
        // World point (input)
        let p = point.0.coords;
        // Camera point (intermediate output)
        let pc = (self.0 * point.0).coords;

        // dP/dT (Jacobian of camera point in respect to translation component)
        let dp_dt = Matrix3::identity();

        // d/dQv (Qv x (Qv x P))
        let qv_qv_p = Matrix3::new(
            q.y * p.y + q.z * p.z,          q.y * p.x - 2.0 * q.x * p.y,    q.z * p.x - 2.0 * q.x * p.z,
            q.x * p.y - 2.0 * q.y * p.x,    q.x * p.x + q.z * p.z,          q.z * p.y - 2.0 * q.y * p.z,
            q.x * p.z - 2.0 * q.z * p.x,    q.y * p.z - 2.0 * q.z * p.y,    q.x * p.x + q.y * p.y
        );
        // d/dQv (Qv x P)
        let qv_p = Matrix3::new(
            0.0,    -p.z,   p.y,
            p.z,    0.0,    -p.x,
            -p.y,   p.x,    0.0,
        );
        // dP/dQv = d/dQv (2 * Qs * Qv x P + 2 * Qv x (Qv x P))
        // Jacobian of camera point in respect to vector component of quaternion
        let dp_dqv = 2.0 * (q.w * qv_p + qv_qv_p);

        // dP/Ds = d/Qs (2 * Qs * Qv x P)
        // Jacobian of camera point in respect to real component of quaternion
        let dp_ds = 2.0 * q.xyz().cross(&p);
        
        // dP/dT,Q (Jacobian of 3d camera point in respect to translation and quaternion)
        let dp_dtq = MatrixMN::<f32, U7, U3>::from_rows(&[
            dp_dt.row(0).into(),
            dp_dt.row(1).into(),
            dp_dt.row(2).into(),
            dp_dqv.row(0).into(),
            dp_dqv.row(1).into(),
            dp_dqv.row(2).into(),
            dp_ds.transpose(),
        ]);
        
        // 1 / pz
        let pcz = pc.z.recip();
        // - 1 / pz^2
        let npcz2 = -(pcz * pcz);
        
        // dK/dp (Jacobian of normalized image coordinate in respect to 3d camera point)
        let dk_dp = Matrix3x2::new(
            pcz,    0.0,
            0.0,    pcz,
            npcz2,  npcz2,
        );

        dp_dtq * dk_dp
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
