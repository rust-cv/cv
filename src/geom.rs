use crate::NormalizedKeyPoint;
use nalgebra::{Vector2, Vector3};

/// This function consumes a `from` `xy` coordinate, a `to` [`NormalizedKeyPoint`], and a direction
/// to translate along.
///
/// - `t` the `translation` vector
/// - `b` the `from` point
/// - `a` the `to` epipolar point
/// - `O` the optical center
/// - `@` the virtual image plane
///
/// ```no_build,no_run
///      t<---b
///          /
///         /
/// @@@a@@@/@@@@@
///    |  /
///    | /
///    |/
///    O
/// ```
///
/// The `from` coordinate is actually the relative 3d coordinate in camera space,
/// but the `z` is truncated because it is not required to compute the translation, otherwise it
/// would be a [`CameraPoint`]. Effectively, `from` is a [`CameraPoint`] without a `z` component.
///
/// The `to` coordinate is just a normalized keypoint that we wish to find the optimal translation
/// to reproject as close as possible to.
///
/// The `translation` is a vector which will be scaled (multiplied) by the return value to
/// get the actual 3d translation to move from `from` to `to` in 3d space.
///
/// See the document `notes/derivation_of_reproject_along_translation.md` in the repository for how this
/// was derived.
pub fn reproject_along_translation(
    from: Vector2<f32>,
    to: NormalizedKeyPoint,
    translation: Vector3<f32>,
) -> f32 {
    let NormalizedKeyPoint(a) = to;
    let b = from;
    let t = translation;
    (a.y * b.x - a.x * b.y) / (a.x * t.y - a.y * t.x)
}
