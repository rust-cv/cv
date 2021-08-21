use derive_more::AsRef;
use nalgebra::{Point3, UnitVector3, Vector4};

#[cfg(feature = "serde-serialize")]
use serde::{Deserialize, Serialize};

/// This trait is implemented for homogeneous projective 3d coordinate.
pub trait Projective: Clone + Copy {
    /// Retrieve the homogeneous vector.
    ///
    /// The homonegeous vector is guaranteed to have xyz normalized.
    /// The distance of the point is encoded as the reciprocal of the `w` component.
    /// The `w` component is guaranteed to be positive or zero.
    /// A `w` component of `0` implies the point is at infinity.
    fn homogeneous(self) -> Vector4<f64>;

    /// Create the projective using a homogeneous vector.
    ///
    /// This will normalize the xyz components of the provided vector and adjust `w` accordingly.
    fn from_homogeneous(mut point: Vector4<f64>) -> Self {
        if point.w.is_sign_negative() {
            point = -point;
        }
        Self::from_homogeneous_unchecked(point.unscale(point.xyz().norm()))
    }

    /// It is not recommended to call this directly, unless you have a good reason.
    ///
    /// The xyz components MUST be of unit length (normalized), and `w` must be adjusted accordingly.
    /// See [`Projective::homogeneous`] for more details.
    fn from_homogeneous_unchecked(point: Vector4<f64>) -> Self;

    /// Retrieve the euclidean 3d point by normalizing the homogeneous coordinate.
    ///
    /// This may fail, as a homogeneous coordinate can exist at near-infinity (like a star in the sky),
    /// whereas a 3d euclidean point cannot (it would overflow).
    fn point(self) -> Option<Point3<f64>> {
        Point3::from_homogeneous(self.homogeneous())
    }

    /// Convert the euclidean 3d point into homogeneous coordinates.
    fn from_point(point: Point3<f64>) -> Self {
        Self::from_homogeneous(point.to_homogeneous())
    }

    /// Retrieve the normalized bearing of the coordinate.
    fn bearing(self) -> UnitVector3<f64> {
        UnitVector3::new_unchecked(self.homogeneous().xyz())
    }
}

/// A 3d point in the camera's reference frame.
///
/// In the camera's reference frame, the origin is the optical center,
/// positive X axis is right, positive Y axis is down, and positive Z axis is forwards.
///
/// The unit of distance of a `CameraPoint` is unspecified, but it should be consistent relative
/// to other points in the reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsRef)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct CameraPoint(Vector4<f64>);

impl Projective for CameraPoint {
    fn homogeneous(self) -> Vector4<f64> {
        self.0
    }

    fn from_homogeneous_unchecked(point: Vector4<f64>) -> Self {
        Self(point)
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
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsRef)]
#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
pub struct WorldPoint(pub Vector4<f64>);

impl Projective for WorldPoint {
    fn homogeneous(self) -> Vector4<f64> {
        self.0
    }

    fn from_homogeneous_unchecked(point: Vector4<f64>) -> Self {
        Self(point)
    }
}
