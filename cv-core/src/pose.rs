use crate::{Bearing, CameraPoint, FeatureWorldMatch, Projective, Skew3, WorldPoint};
use derive_more::{AsMut, AsRef, From, Into};
use nalgebra::{
    IsometryMatrix3, Matrix4, Matrix4x6, Matrix6x4, Rotation3, Vector3, Vector4, Vector6,
};
use sample_consensus::Model;

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

/// This trait is implemented by all the different poses in this library:
///
/// * [`CameraToWorld`] - Transforms [`CameraPoint`] into [`WorldPoint`]
/// * [`WorldToCamera`] - Transforms [`WorldPoint`] into [`CameraPoint`]
/// * [`CameraToCamera`] - Transforms [`CameraPoint`] from one camera into [`CameraPoint`] for another camera
pub trait Pose: From<IsometryMatrix3<f64>> + Clone + Copy {
    type InputPoint: Projective;
    type OutputPoint: Projective;
    type Inverse: Pose;

    /// Retrieve the isometry.
    fn isometry(self) -> IsometryMatrix3<f64>;

    /// Creates a pose with no change in position or orientation.
    fn identity() -> Self {
        IsometryMatrix3::identity().into()
    }

    /// Takes the inverse of the pose.
    fn inverse(self) -> Self::Inverse {
        self.isometry().inverse().into()
    }

    /// Applies a scale factor to the pose (scales the translation component)
    fn scale(self, scale: f64) -> Self {
        let mut isometry = self.isometry();
        isometry.translation.vector *= scale;
        isometry.into()
    }

    /// Create the pose from rotation and translation.
    fn from_parts(translation: Vector3<f64>, rotation: Rotation3<f64>) -> Self {
        IsometryMatrix3::from_parts(translation.into(), rotation).into()
    }

    /// Retrieve the homogeneous matrix.
    fn homogeneous(self) -> Matrix4<f64> {
        self.isometry().to_homogeneous()
    }

    /// Retrieve the se(3) representation of the pose.
    fn se3(self) -> Vector6<f64> {
        let isometry = self.isometry();
        let t = isometry.translation.vector;
        let r: Skew3 = isometry.rotation.into();
        Vector6::new(t.x, t.y, t.z, r.x, r.y, r.z)
    }

    /// Set the se(3) representation of the pose.
    fn from_se3(se3: Vector6<f64>) -> Self {
        let translation = se3.xyz();
        let rotation = Skew3(Vector3::new(se3[3], se3[4], se3[5])).into();
        Self::from_parts(translation, rotation)
    }

    /// Transform the given point to an output point, while also retrieving both Jacobians.
    ///
    /// The following things are returned in this order:
    ///
    /// * The output point of the transformation
    /// * The Jacobian of the output in respect to the input point
    /// * The Jacobian of the output in respect to the pose in se(3) (with translation components before so(3) components)
    fn transform_jacobians(
        self,
        input: Self::InputPoint,
    ) -> (Self::OutputPoint, Matrix4<f64>, Matrix4x6<f64>) {
        let (rotated, output) = pose_rotated_output(self, input);
        let jacobian_input = pose_jacobian_input(self);
        let jacobian_self = pose_jacobian_self(self, rotated, output);
        (output.into(), jacobian_input, jacobian_self)
    }

    /// Transform the given point to an output point, while also retrieving the input Jacobian.
    ///
    /// The following things are returned in this order:
    ///
    /// * The output point of the transformation
    /// * The Jacobian of the output in respect to the input point
    fn transform_jacobian_input(
        self,
        input: Self::InputPoint,
    ) -> (Self::OutputPoint, Matrix4<f64>) {
        let output = pose_output(self, input);
        let jacobian_input = pose_jacobian_input(self);
        (output.into(), jacobian_input)
    }

    /// Transform the given point to an output point, while also retrieving the transform Jacobian.
    ///
    /// The following things are returned in this order:
    ///
    /// * The output point of the transformation
    /// * The Jacobian of the output in respect to the pose in se(3) (with translation components before so(3) components)
    fn transform_jacobian_self(
        self,
        input: Self::InputPoint,
    ) -> (Self::OutputPoint, Matrix4x6<f64>) {
        let (rotated, output) = pose_rotated_output(self, input);
        let jacobian_self = pose_jacobian_self(self, rotated, output);
        (output.into(), jacobian_self)
    }

    /// Transform the given point to an output point, while also retrieving the transform Jacobian.
    ///
    /// The following things are returned in this order:
    ///
    /// * The output point of the transformation
    /// * The Jacobian of the output in respect to the pose in se(3) (with translation components before so(3) components)
    fn transform(self, input: Self::InputPoint) -> Self::OutputPoint {
        pose_output(self, input).into()
    }
}

/// Retrieves the output coordinate from the pose and input.
fn pose_output<P: Pose>(pose: P, input: P::InputPoint) -> Vector4<f64>
where
    P::InputPoint: From<Vector4<f64>>,
{
    pose.isometry().to_homogeneous() * input.homogeneous()
}

/// Retrieves the rotated and output coordinates (in that order) from the pose and input.
fn pose_rotated_output<P: Pose>(pose: P, input: P::InputPoint) -> (Vector4<f64>, Vector4<f64>)
where
    P::InputPoint: From<Vector4<f64>>,
{
    let rotated = pose.isometry().rotation.to_homogeneous() * input.homogeneous();
    let output = pose.isometry().to_homogeneous() * input.homogeneous();
    (rotated, output)
}

/// Retrieves the Jacobian relating the output to the input.
fn pose_jacobian_input<P: Pose>(pose: P) -> Matrix4<f64> {
    pose.isometry().to_homogeneous()
}

/// Retrieves the Jacobian relating the output to the pose in se(3)
fn pose_jacobian_self<P: Pose>(
    pose: P,
    rotated: Vector4<f64>,
    output: Vector4<f64>,
) -> Matrix4x6<f64> {
    // The translation homogeneous matrix
    //
    // This is also the jacobian of the output in respect to the rotation output.
    let translation = pose.isometry().translation.to_homogeneous();

    // dP/dT (Jacobian of output point in respect to translation component)
    let dp_dt = Matrix4::<f64>::identity() * output.w;

    // dP/ds (Jacobian of output point in respect to skew component)
    let dp_ds = translation * Skew3::jacobian_self(rotated.xyz()).to_homogeneous();

    // dP/dT,s (Jacobian of 3d camera point in respect to translation and skew)
    Matrix6x4::<f64>::from_rows(&[
        dp_dt.row(0),
        dp_dt.row(1),
        dp_dt.row(2),
        dp_ds.row(0),
        dp_ds.row(1),
        dp_ds.row(2),
    ])
    .transpose()
}

/// This contains a world pose, which is a pose of the world relative to the camera.
/// This maps [`WorldPoint`] into [`CameraPoint`], changing an absolute position into
/// a vector relative to the camera.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, From, Into)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct WorldToCamera(pub IsometryMatrix3<f64>);

impl Pose for WorldToCamera {
    type InputPoint = WorldPoint;
    type OutputPoint = CameraPoint;
    type Inverse = CameraToWorld;

    fn isometry(self) -> IsometryMatrix3<f64> {
        self.into()
    }
}

impl<P> Model<FeatureWorldMatch<P>> for WorldToCamera
where
    P: Bearing,
{
    fn residual(&self, data: &FeatureWorldMatch<P>) -> f64 {
        let WorldToCamera(iso) = *self;
        let FeatureWorldMatch(feature, world) = data;

        let new_bearing = (iso.to_homogeneous() * world.0).xyz().normalize();
        let bearing_vector = feature.bearing();
        1.0 - bearing_vector.dot(&new_bearing)
    }
}

/// This contains a camera pose, which is a pose of the camera relative to the world.
/// This transforms camera points (with depth as `z`) into world coordinates.
/// This also tells you where the camera is located and oriented in the world.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, From, Into)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct CameraToWorld(pub IsometryMatrix3<f64>);

impl Pose for CameraToWorld {
    type InputPoint = CameraPoint;
    type OutputPoint = WorldPoint;
    type Inverse = WorldToCamera;

    fn isometry(self) -> IsometryMatrix3<f64> {
        self.into()
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
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, From, Into)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct CameraToCamera(pub IsometryMatrix3<f64>);

impl Pose for CameraToCamera {
    type InputPoint = CameraPoint;
    type OutputPoint = CameraPoint;
    type Inverse = CameraToCamera;

    fn isometry(self) -> IsometryMatrix3<f64> {
        self.into()
    }
}
