use derive_more::{AsMut, AsRef, Deref, DerefMut, From, Into};
use nalgebra::{Point3, Unit, Vector3, Vector4};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

/// This trait is implemented for homogeneous projective 3d coordinate.
pub trait Projective: From<Vector4<f64>> + Clone + Copy {
    /// Retrieve the homogeneous vector.
    ///
    /// No constraints are put on this vector. All components can move freely and it is not normalized.
    /// However, this vector may be normalized if desired and it will still be equivalent to the original.
    /// You may wish to normalize it if you want to avoid floating point precision issues, for instance.
    fn homogeneous(self) -> Vector4<f64>;

    /// Retrieve the euclidean 3d point by normalizing the homogeneous coordinate.
    ///
    /// This may fail, as a homogeneous coordinate can exist at near-infinity (like a star in the sky),
    /// whereas a 3d euclidean point cannot (it would overflow).
    fn point(self) -> Option<Point3<f64>> {
        Point3::from_homogeneous(self.homogeneous())
    }

    /// Convert the euclidean 3d point into homogeneous coordinates.
    fn from_point(point: Point3<f64>) -> Self {
        point.to_homogeneous().into()
    }

    /// Retrieve the normalized bearing of the coordinate.
    fn bearing(self) -> Unit<Vector3<f64>> {
        Unit::new_normalize(self.bearing_unnormalized())
    }

    /// Retrieve the unnormalized bearing of the coordinate.
    ///
    /// Use this when you know that you do not need the bearing to be normalized,
    /// and it may increase performance. Otherwise use [`Projective::bearing`].
    fn bearing_unnormalized(self) -> Vector3<f64> {
        self.homogeneous().xyz()
    }
}

/// A 3d point which is relative to the camera's optical center and orientation where
/// the positive X axis is right, positive Y axis is down, and positive Z axis is forwards
/// from the optical center of the camera. The unit of distance of a `CameraPoint` is
/// unspecified and relative to the current reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsMut, AsRef, Deref, DerefMut, From, Into)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct CameraPoint(pub Vector4<f64>);

impl Projective for CameraPoint {
    fn homogeneous(self) -> Vector4<f64> {
        self.into()
    }
}

/// A point in "world" coordinates.
/// This means that the real-world units of the pose are unknown, but the
/// unit of distance and orientation are the same as the current reconstruction.
///
/// The reason that the unit of measurement is typically unknown is because if
/// the whole world is scaled by any factor `n` (excluding the camera itself), then
/// the normalized image coordinates will be exactly same on every frame. Due to this,
/// the scaling of the world is chosen arbitrarily.
///
/// To extract the real scale of the world, a known distance between two `WorldPoint`s
/// must be used to scale the whole world (and all translations between cameras). At
/// that point, the world will be appropriately scaled. It is recommended not to make
/// the `WorldPoint` in the reconstruction scale to the "correct" scale. This is for
/// two reasons:
///
/// Firstly, because it is possible for scale drift to occur due to the above situation,
/// the further in the view graph you go from the reference measurement, the more the scale
/// will drift from the reference. It would give a false impression that the scale is known
/// globally when it is only known locally if the whole reconstruction was scaled.
///
/// Secondly, as the reconstruction progresses, the reference points might get rescaled
/// as optimization of the reconstruction brings everything into global consistency.
/// This means that, while the reference points would be initially scaled correctly,
/// any graph optimization might cause them to drift in scale as well.
///
/// Please scale your points on-demand. When you need to know a real distance in the
/// reconstruction, please use the closest known refenence in the view graph to scale
/// it appropriately. In the future we will add APIs to utilize references
/// as optimization constraints when a known reference reconstruction is present.
///
/// If you must join two reconstructions, please solve for the similarity (rotation, translation and scale)
/// between the two reconstructions using an optimizer. APIs will eventually be added to perform this operation
/// as well.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsMut, AsRef, Deref, DerefMut, From, Into)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct WorldPoint(pub Vector4<f64>);

impl Projective for WorldPoint {
    fn homogeneous(self) -> Vector4<f64> {
        self.into()
    }
}
