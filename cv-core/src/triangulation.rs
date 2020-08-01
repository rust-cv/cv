use crate::{Bearing, CameraPoint, CameraToCamera, Pose, WorldPoint, WorldToCamera};

/// This trait is for algorithms which allow you to triangulate a point from two or more observances.
/// Each observance is a [`WorldToCamera`] and a [`Bearing`].
pub trait TriangulatorObservations {
    fn triangulate_observations<B: Bearing>(
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
    T: TriangulatorObservations,
{
    fn triangulate_relative<A: Bearing, B: Bearing>(
        &self,
        CameraToCamera(pose): CameraToCamera,
        a: A,
        b: B,
    ) -> Option<CameraPoint> {
        use core::iter::once;

        // We use the first camera as the "world".
        // The first pose maps the first camera to itself (the world).
        // The second pose maps the first camera (the world) to the second camera (the camera).
        // This is how we convert the `CameraToCamera` into a `WorldToCamera`.
        self.triangulate_observations(
            once((WorldToCamera::identity(), a.bearing()))
                .chain(once((WorldToCamera(pose), b.bearing()))),
        )
        .map(|p| CameraPoint(p.0))
    }
}
