use crate::{Bearing, CameraPoint, CameraToCamera, Pose, WorldPoint, WorldToCamera};
use nalgebra::{IsometryMatrix3, Vector3};

/// This trait is for algorithms which allow you to triangulate a point from two or more observances.
/// Each observance is a [`WorldToCamera`] and a [`Bearing`].
pub trait TriangulatorObservances {
    fn triangulate_observances<B: Bearing>(
        &self,
        pairs: impl IntoIterator<Item = (WorldToCamera, B)>,
    ) -> Option<WorldPoint>;
}

/// This trait allows you to take one relative pose from camera `A` to camera `B` and two bearings `a` and `b` from
/// their respective cameras to triangulate a point from the perspective of camera `A`.
pub trait TriangulatorRelative {
    fn triangulate_relative<A: Bearing, B: Bearing>(
        &self,
        relative_pose: CameraToCamera,
        a: A,
        b: B,
    ) -> Option<CameraPoint>;
}

impl<T> TriangulatorRelative for T
where
    T: TriangulatorObservances,
{
    fn triangulate_relative<A: Bearing, B: Bearing>(
        &self,
        CameraToCamera(pose): CameraToCamera,
        a: A,
        b: B,
    ) -> Option<CameraPoint> {
        use core::iter::once;

        self.triangulate_observances(
            once((WorldToCamera::identity(), a.bearing()))
                .chain(once((WorldToCamera(pose), b.bearing()))),
        )
        .map(|p| CameraPoint(p.0))
    }
}

/// This trait allows you to project the point `a` onto the bearing `b` by only scaling a translation vector `t`.
/// The returned value is the amout to scale `t` by to achieve the triangulation.
/// All inputs share the same origin (optical center). Below is a visualization of the problem.
///
/// - `t` the translation vector that needs to be scaled
/// - `a` the source point which is relative to the camera (see [`CameraPoint`])
/// - `b` the destination bearing
/// - `O` the optical center
/// - `@` the virtual image plane
///
/// ```text
///      t<---a
///    ^     /
///    |    /
/// @@@b@@@/@@@@@
///    |  /
///    | /
///    |/
///    O
/// ```
pub trait TriangulatorProject {
    fn triangulate_project<B: Bearing>(
        &self,
        from: CameraPoint,
        onto: B,
        translation: Vector3<f64>,
    ) -> Option<f64>;
}

impl<T> TriangulatorProject for T
where
    T: TriangulatorRelative,
{
    fn triangulate_project<B: Bearing>(
        &self,
        from: CameraPoint,
        onto: B,
        translation: Vector3<f64>,
    ) -> Option<f64> {
        // Create a fake relative camera pose which is looking at the point.
        let eye = from.0;
        let target = eye + translation;
        let up = from.coords.cross(&translation);
        let fake_pose = CameraToCamera(IsometryMatrix3::look_at_lh(&eye, &target, &up));
        self.triangulate_relative(fake_pose, onto, Vector3::z_axis())
            .map(|point| {
                // Get the vector representing the difference from `from`.
                let delta = point.0 - eye;
                // Normalize the translation.
                let dir = translation.normalize();
                // Take the dot product to see how far along the translation the point is.
                delta.dot(&dir)
            })
    }
}
