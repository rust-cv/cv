use crate::{KeyPointWorldMatch, NormalizedKeyPoint, WorldPoint, CameraPoint};
use derive_more::{AsMut, AsRef, Constructor, Deref, DerefMut, From, Into};
use nalgebra::{
    dimension::{U2, U3, U7},
    Isometry3, Matrix3, Matrix3x2, MatrixMN, Quaternion, Translation3, UnitQuaternion, Vector2,
    Vector3, Vector4, VectorN,
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
    pub fn project(&self, point: WorldPoint) -> NormalizedKeyPoint {
        self.project_camera(point).into()
    }

    /// Projects the [`WorldPoint`] into camera space as a [`CameraPoint`].
    pub fn project_camera(&self, WorldPoint(point): WorldPoint) -> CameraPoint {
        let WorldPose(iso) = *self;
        CameraPoint((iso * point).coords)
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

    pub fn to_vec(&self) -> VectorN<f32, U7> {
        let Self(iso) = *self;
        let t = iso.translation.vector;
        let rc = iso.rotation.quaternion().coords;
        t.push(rc.x).push(rc.y).push(rc.z).push(rc.w)
    }

    pub fn from_vec(v: VectorN<f32, U7>) -> Self {
        Self(Isometry3::from_parts(
            Translation3::from(Vector3::new(v[0], v[1], v[2])),
            UnitQuaternion::from_quaternion(Quaternion::from(Vector4::new(v[3], v[4], v[5], v[6]))),
        ))
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

/// This stores an essential matrix, which is satisfied by the following constraint:
/// 
/// transpose(x') * E * x = 0
/// 
/// Where `x'` and `x` are homogeneous normalized image coordinates. You can get a
/// homogeneous normalized image coordinate by appending `1.0` to a `NormalizedKeyPoint`.
/// 
/// The essential matrix embodies the epipolar constraint between two images. Given that light
/// travels in a perfectly straight line (it will not, but for short distances it mostly does)
/// and assuming a pinhole camera model, for any point on the camera sensor, the light source
/// for that point exists somewhere along a line extending out from the bearing (direction
/// of travel) of that point. For a normalized image coordinate, that bearing is `(x, y, 1.0)`.
/// That is because normalized image coordinates exist on a virtual plane (the sensor)
/// a distance `z = 1.0` from the optical center (the location of the focal point) where
/// the unit of distance is the focal length. In epipolar geometry, the point on the virtual
/// plane is called an epipole. The line through 3d space created by the bearing that travels
/// from the optical center through the epipole is called an epipolar line.
/// 
/// If you look at every point along an epipolar line, each one of those points would show
/// up as a different point on the camera sensor of another image (if they are in view).
/// If you traced every point along this epipolar line to where it would appear on the sensor
/// of the camera (projection of the 3d points into normalized image coordinates), then
/// the points would form a straight line. This means that you can draw epipolar lines
/// that do not pass through the optical center of an image on that image.
/// 
/// The essential matrix makes it possible to create a vector that is perpendicular to all
/// bearings that are formed from the epipolar line on the second image's sensor. This is
/// done by computing `E * x`, where `x` is a homogeneous normalized image coordinate
/// from the first image. The transpose of the resulting vector then has a dot product
/// with the transpose of the second image coordinate `x'` which is equal to `0.0`.
/// This can be written as:
/// 
/// ```no_build,no_run
/// dot(transpose(E * x), x') = 0
/// ```
/// 
/// This can be re-written into the form given above:
/// 
/// ```no_build,no_run
/// transpose(x') * E * x = 0
/// ```
/// 
/// Where the first operation creates a pependicular vector to the epipoles on the first image
/// and the second takes the dot product which should result in 0.
/// 
/// With a `EssentialMatrix`, you can retrieve the rotation and translation given
/// one normalized image coordinate and one bearing that is scaled to the depth
/// of the point relative to the current reconstruction. This kind of point can be computed
/// using [`WorldPose::project_camera`] to convert a [`WorldPoint`] to a [`CameraPoint`].
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
struct EssentialMatrix(pub Matrix3<f32>);
