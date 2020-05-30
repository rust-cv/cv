use crate::{Bearing, CameraPoint, EssentialMatrix, FeatureWorldMatch, Skew3, WorldPoint};
use derive_more::{AsMut, AsRef, Deref, DerefMut, From, Into};
use nalgebra::{IsometryMatrix3, Matrix3, Matrix3x2, Matrix6x2, Matrix6x3, Rotation3, Vector3};
use sample_consensus::Model;

/// This contains a world pose, which is a pose of the world relative to the camera.
/// This maps [`WorldPoint`] into [`CameraPoint`], changing an absolute position into
/// a vector relative to the camera.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct WorldPose(pub IsometryMatrix3<f64>);

impl<P> Model<FeatureWorldMatch<P>> for WorldPose
where
    P: Bearing,
{
    fn residual(&self, data: &FeatureWorldMatch<P>) -> f32 {
        let WorldPose(iso) = *self;
        let FeatureWorldMatch(feature, world) = data;

        let new_bearing = (iso * world.coords).normalize();
        let bearing_vector = feature.bearing();
        (1.0 - bearing_vector.dot(&new_bearing)) as f32
    }
}

impl WorldPose {
    /// Projects the [`WorldPoint`] into camera space as a [`CameraPoint`].
    pub fn transform(&self, WorldPoint(point): WorldPoint) -> CameraPoint {
        let WorldPose(iso) = *self;
        CameraPoint(iso * point)
    }

    /// Computes the Jacobian of the projection in respect to the `WorldPose`.
    ///
    /// The Jacobian is in the format:
    /// ```no_build,no_run
    /// | dx/dtx dy/dPx |
    /// | dx/dty dy/dPy |
    /// | dx/dtz dy/dPz |
    /// | dx/dwx dy/dwx |
    /// | dx/dwy dy/dwy |
    /// | dx/dwz dy/dwz |
    /// ```
    ///
    /// Where `t` refers to the translation vector and
    /// `w` refers to the log map of the rotation in so(3).
    #[rustfmt::skip]
    pub fn projection_pose_jacobian(&self, point: WorldPoint) -> Matrix6x2<f64> {
        // Rotated point (intermediate output)
        let pr = (self.0.rotation * point.0).coords;
        // Camera point/rotated and translated point (intermediate output)
        let pc = (self.0 * point.0).coords;

        // dP/dT (Jacobian of camera point in respect to translation component)
        let dp_dt = Matrix3::<f64>::identity();

        // dP/dR
        let dp_dr = Skew3::jacobian_output_to_self(pr);

        // dP/dT,Q (Jacobian of 3d camera point in respect to translation and quaternion)
        let dp_dtq = Matrix6x3::<f64>::from_rows(&[
            dp_dt.row(0),
            dp_dt.row(1),
            dp_dt.row(2),
            dp_dr.row(0),
            dp_dr.row(1),
            dp_dr.row(2),
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
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct CameraPose(pub IsometryMatrix3<f64>);

impl CameraPose {
    pub fn identity() -> Self {
        Self(IsometryMatrix3::identity())
    }
}

impl From<WorldPose> for CameraPose {
    fn from(world: WorldPose) -> Self {
        Self(world.inverse())
    }
}

/// This contains a relative pose, which is a pose that transforms the [`CameraPoint`]
/// of one image into the corresponding [`CameraPoint`] of another image. This transforms
/// the point from the camera space of camera `A` to camera `B`.
///
/// Camera space for a given camera is defined as thus:
///
/// * Origin is the optical center
/// * Positive z axis is forwards
/// * Positive y axis is up
/// * Positive x axis is right
///
/// Note that this is a left-handed coordinate space.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct RelativeCameraPose(pub IsometryMatrix3<f64>);

impl RelativeCameraPose {
    /// Create the pose from rotation and translation.
    pub fn from_parts(rotation: Rotation3<f64>, translation: Vector3<f64>) -> Self {
        Self(IsometryMatrix3::from_parts(translation.into(), rotation))
    }

    /// The relative pose transforms the point in camera space from camera `A` to camera `B`.
    pub fn transform(&self, CameraPoint(point): CameraPoint) -> CameraPoint {
        let Self(iso) = *self;
        CameraPoint(iso * point)
    }

    /// Generates an essential matrix corresponding to this relative camera pose.
    ///
    /// If a point `a` is transformed using [`RelativeCameraPose::transform`] into
    /// a point `b`, then the essential matrix returned by this method will
    /// give a residual of approximately `0.0` when you call
    /// `essential.residual(&FeatureMatch(a, b))`.
    ///
    /// See the documentation of [`EssentialMatrix`] for more information.
    pub fn essential_matrix(&self) -> EssentialMatrix {
        EssentialMatrix(self.0.translation.vector.cross_matrix() * *self.0.rotation.matrix())
    }
}

/// This stores a [`RelativeCameraPose`] that has not had its translation scaled.
///
/// The translation for an unscaled relative camera pose should allow the
/// triangulation of correspondences to lie in front of both cameras.
/// Aside from that case, the relative pose contained inside should only
/// be used to initialize a reconstruction with unknown scale.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct UnscaledRelativeCameraPose(pub RelativeCameraPose);
