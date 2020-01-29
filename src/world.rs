use derive_more::{AsMut, AsRef, Deref, DerefMut, From, Into};
use nalgebra::Point3;

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
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct WorldPoint(pub Point3<f64>);
