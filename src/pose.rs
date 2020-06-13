use crate::{Bearing, CameraPoint, EssentialMatrix, FeatureWorldMatch, Skew3, WorldPoint};
use derive_more::{AsMut, AsRef, Deref, DerefMut, From, Into};
use nalgebra::{IsometryMatrix3, Matrix3, Matrix6x3, Point3, Rotation3, Vector3, Vector6};
use sample_consensus::Model;

pub trait Pose {
    type InputPoint;
    type OutputPoint;

    /// Creates a pose with no change in position or orientation.
    fn identity() -> Self;

    /// Create the pose from rotation and translation.
    fn from_parts(translation: Vector3<f64>, rotation: Rotation3<f64>) -> Self;

    /// Transform the given point to an output point, while also retrieving both Jacobians.
    ///
    /// The following things are returned in this order:
    ///
    /// * The output point of the transformation
    /// * The Jacobian of the output in respect to the input point
    /// * The Jacobian of the output in respect to the pose in se(3) (with translation components before so(3) components)
    fn transform_jacobians(
        &self,
        input: Self::InputPoint,
    ) -> (Self::OutputPoint, Matrix3<f64>, Matrix6x3<f64>);

    /// Transform the given point to an output point, while also retrieving the input Jacobian.
    ///
    /// The following things are returned in this order:
    ///
    /// * The output point of the transformation
    /// * The Jacobian of the output in respect to the input point
    fn transform_jacobian_input(
        &self,
        input: Self::InputPoint,
    ) -> (Self::OutputPoint, Matrix3<f64>) {
        let (output, input_jacobian, _) = self.transform_jacobians(input);
        (output, input_jacobian)
    }

    /// Transform the given point to an output point, while also retrieving the transform Jacobian.
    ///
    /// The following things are returned in this order:
    ///
    /// * The output point of the transformation
    /// * The Jacobian of the output in respect to the pose in se(3) (with translation components before so(3) components)
    fn transform_jacobian_pose(
        &self,
        input: Self::InputPoint,
    ) -> (Self::OutputPoint, Matrix6x3<f64>) {
        let (output, _, pose_jacobian) = self.transform_jacobians(input);
        (output, pose_jacobian)
    }

    /// Transform the given point to an output point, while also retrieving the transform Jacobian.
    ///
    /// The following things are returned in this order:
    ///
    /// * The output point of the transformation
    /// * The Jacobian of the output in respect to the pose in se(3) (with translation components before so(3) components)
    fn transform(&self, input: Self::InputPoint) -> Self::OutputPoint {
        let (output, _, _) = self.transform_jacobians(input);
        output
    }

    /// Retrieve the se(3) representation of the pose.
    fn se3(&self) -> Vector6<f64>;

    /// Set the se(3) representation of the pose.
    fn set_se3(&mut self, se3: Vector6<f64>);
}

/// Transform the given point while also retrieving both Jacobians.
///
/// The following things are returned in this order:
///
/// * The output point of the transformation
/// * The Jacobian of the output in respect to the input point
/// * The Jacobian of the output in respect to the pose in se(3) (with translation components before so(3) components)
#[inline]
fn isometry_point_outputs(
    isometry: &IsometryMatrix3<f64>,
    input: Point3<f64>,
) -> (Point3<f64>, Matrix3<f64>, Matrix6x3<f64>) {
    // Rotated point (intermediate output)
    let rotated = isometry.rotation * input;
    // Totally transfored output
    let output = rotated + isometry.translation.vector;

    // dP/dT (Jacobian of camera point in respect to translation component)
    let dp_dt = Matrix3::<f64>::identity();

    // dP/ds (Jacobian of output point in respect to skew component)
    let dp_ds = Skew3::jacobian_self(rotated.coords);

    // dP/dT,s (Jacobian of 3d camera point in respect to translation and skew)
    let dp_dts = Matrix6x3::<f64>::from_rows(&[
        dp_dt.row(0),
        dp_dt.row(1),
        dp_dt.row(2),
        dp_ds.row(0),
        dp_ds.row(1),
        dp_ds.row(2),
    ]);

    // Turn the rotation into a skew to extract the input jacobian.
    let skew: Skew3 = isometry.rotation.into();
    let dp_di = skew.jacobian_input();

    (output, dp_di, dp_dts)
}

/// This contains a world pose, which is a pose of the world relative to the camera.
/// This maps [`WorldPoint`] into [`CameraPoint`], changing an absolute position into
/// a vector relative to the camera.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct WorldToCamera(pub IsometryMatrix3<f64>);

impl Pose for WorldToCamera {
    type InputPoint = WorldPoint;
    type OutputPoint = CameraPoint;

    fn identity() -> Self {
        Self(IsometryMatrix3::identity())
    }

    fn from_parts(translation: Vector3<f64>, rotation: Rotation3<f64>) -> Self {
        Self(IsometryMatrix3::from_parts(translation.into(), rotation))
    }

    #[inline]
    fn transform_jacobians(
        &self,
        input: Self::InputPoint,
    ) -> (Self::OutputPoint, Matrix3<f64>, Matrix6x3<f64>) {
        let (output, jacobian_input, jacobian_pose) = isometry_point_outputs(&self.0, input.0);
        (CameraPoint(output), jacobian_input, jacobian_pose)
    }

    #[inline]
    fn se3(&self) -> Vector6<f64> {
        let t = self.translation.vector;
        let r: Skew3 = self.rotation.into();
        Vector6::new(t.x, t.y, t.z, r.x, r.y, r.z)
    }

    #[inline]
    fn set_se3(&mut self, se3: Vector6<f64>) {
        self.translation.vector = se3.xyz();
        self.rotation = Skew3(Vector3::new(se3[3], se3[4], se3[5])).into();
    }
}

impl<P> Model<FeatureWorldMatch<P>> for WorldToCamera
where
    P: Bearing,
{
    fn residual(&self, data: &FeatureWorldMatch<P>) -> f32 {
        let WorldToCamera(iso) = *self;
        let FeatureWorldMatch(feature, world) = data;

        let new_bearing = (iso * world.coords).normalize();
        let bearing_vector = feature.bearing();
        (1.0 - bearing_vector.dot(&new_bearing)) as f32
    }
}

impl From<CameraToWorld> for WorldToCamera {
    fn from(camera: CameraToWorld) -> Self {
        Self(camera.inverse())
    }
}

/// This contains a camera pose, which is a pose of the camera relative to the world.
/// This transforms camera points (with depth as `z`) into world coordinates.
/// This also tells you where the camera is located and oriented in the world.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct CameraToWorld(pub IsometryMatrix3<f64>);

impl Pose for CameraToWorld {
    type InputPoint = CameraPoint;
    type OutputPoint = WorldPoint;

    fn identity() -> Self {
        Self(IsometryMatrix3::identity())
    }

    fn from_parts(translation: Vector3<f64>, rotation: Rotation3<f64>) -> Self {
        Self(IsometryMatrix3::from_parts(translation.into(), rotation))
    }

    #[inline]
    fn transform_jacobians(
        &self,
        input: Self::InputPoint,
    ) -> (Self::OutputPoint, Matrix3<f64>, Matrix6x3<f64>) {
        let (output, jacobian_input, jacobian_pose) = isometry_point_outputs(&self.0, input.0);
        (WorldPoint(output), jacobian_input, jacobian_pose)
    }

    #[inline]
    fn se3(&self) -> Vector6<f64> {
        let t = self.translation.vector;
        let r: Skew3 = self.rotation.into();
        Vector6::new(t.x, t.y, t.z, r.x, r.y, r.z)
    }

    #[inline]
    fn set_se3(&mut self, se3: Vector6<f64>) {
        self.translation.vector = se3.xyz();
        self.rotation = Skew3(Vector3::new(se3[3], se3[4], se3[5])).into();
    }
}

impl From<WorldToCamera> for CameraToWorld {
    fn from(world: WorldToCamera) -> Self {
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
pub struct CameraToCamera(pub IsometryMatrix3<f64>);

impl CameraToCamera {
    /// Generates an essential matrix corresponding to this relative camera pose.
    ///
    /// If a point `a` is transformed using [`Pose::transform`] into
    /// a point `b`, then the essential matrix returned by this method will
    /// give a residual of approximately `0.0` when you call
    /// `essential.residual(&FeatureMatch(a, b))`.
    ///
    /// See the documentation of [`EssentialMatrix`] for more information.
    pub fn essential_matrix(&self) -> EssentialMatrix {
        EssentialMatrix(self.0.translation.vector.cross_matrix() * *self.0.rotation.matrix())
    }
}

impl Pose for CameraToCamera {
    type InputPoint = CameraPoint;
    type OutputPoint = CameraPoint;

    fn identity() -> Self {
        Self(IsometryMatrix3::identity())
    }

    fn from_parts(translation: Vector3<f64>, rotation: Rotation3<f64>) -> Self {
        Self(IsometryMatrix3::from_parts(translation.into(), rotation))
    }

    #[inline]
    fn transform_jacobians(
        &self,
        input: Self::InputPoint,
    ) -> (Self::OutputPoint, Matrix3<f64>, Matrix6x3<f64>) {
        let (output, jacobian_input, jacobian_pose) = isometry_point_outputs(&self.0, input.0);
        (CameraPoint(output), jacobian_input, jacobian_pose)
    }

    #[inline]
    fn se3(&self) -> Vector6<f64> {
        let t = self.translation.vector;
        let r: Skew3 = self.rotation.into();
        Vector6::new(t.x, t.y, t.z, r.x, r.y, r.z)
    }

    #[inline]
    fn set_se3(&mut self, se3: Vector6<f64>) {
        self.translation.vector = se3.xyz();
        self.rotation = Skew3(Vector3::new(se3[3], se3[4], se3[5])).into();
    }
}

/// This stores a [`CameraToCamera`] that has not had its translation scaled.
///
/// The translation for an unscaled camera-to-camera pose should allow the
/// triangulation of correspondences to lie in front of both cameras.
/// Aside from that case, the relative pose contained inside should only
/// be used to initialize a reconstruction with unknown scale.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct UnscaledCameraToCamera(pub IsometryMatrix3<f64>);

impl UnscaledCameraToCamera {
    /// Creates an identity pose with no rotation or translation.
    pub fn identity() -> Self {
        Self(IsometryMatrix3::identity())
    }

    /// Applies the scaling to the unscaled pose
    pub fn scale(mut self, scale: f64) -> CameraToCamera {
        self.translation.vector *= scale;
        CameraToCamera(self.0)
    }

    /// Assume the pose is scaled
    pub fn assume_scaled(self) -> CameraToCamera {
        CameraToCamera(self.0)
    }

    /// Creates the UnscaledCameraToCamera pose from translation and rotation components.
    pub fn from_parts(translation: Vector3<f64>, rotation: Rotation3<f64>) -> Self {
        Self(IsometryMatrix3::from_parts(translation.into(), rotation))
    }
}
