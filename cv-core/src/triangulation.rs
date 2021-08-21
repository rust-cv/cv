use crate::{CameraPoint, CameraToCamera, Pose, Projective, WorldPoint, WorldToCamera};
use nalgebra::UnitVector3;

/// This trait is for algorithms which allow you to triangulate a point from two or more observances.
/// Each observance is a [`WorldToCamera`] and a bearing.
///
/// Returned points will always be checked successfully for chirality.
pub trait TriangulatorObservations {
    /// This function takes a series of [`WorldToCamera`] and bearings that (supposedly) correspond to the same 3d point.
    /// It returns the triangulated [`WorldPoint`] if successful.
    fn triangulate_observations(
        &self,
        pairs: impl Iterator<Item = (WorldToCamera, UnitVector3<f64>)> + Clone,
    ) -> Option<WorldPoint>;

    /// This function takes one bearing (`center_bearing`) coming from the camera whos reference frame we will
    /// triangulate the [`CameraPoint`] in and an iterator over a series of observations
    /// from other cameras, along with the transformation from the original camera to the observation's camera.
    /// It returns the triangulated [`CameraPoint`] if successful.
    #[inline(always)]
    fn triangulate_observations_to_camera(
        &self,
        center_bearing: UnitVector3<f64>,
        pairs: impl Iterator<Item = (CameraToCamera, UnitVector3<f64>)> + Clone,
    ) -> Option<CameraPoint> {
        use core::iter::once;

        // We use the first camera as the "world", thus it is the identity (optical center at origin).
        // The each subsequent pose maps the first camera (the world) to the second camera (the camera).
        // This is how we convert the `CameraToCamera` into a `WorldToCamera`.
        self.triangulate_observations(
            once((WorldToCamera::identity(), center_bearing))
                .chain(pairs.map(|(pose, bearing)| (WorldToCamera(pose.0), bearing))),
        )
        .map(|p| CameraPoint::from_homogeneous(p.0))
    }
}

/// This trait allows you to take one relative pose from camera `A` to camera `B` and two bearings `a` and `b` from
/// their respective cameras to triangulate a point from the perspective of camera `A`.
///
/// Returned points will always be checked successfully for chirality.
pub trait TriangulatorRelative {
    fn triangulate_relative(
        &self,
        relative_pose: CameraToCamera,
        a: UnitVector3<f64>,
        b: UnitVector3<f64>,
    ) -> Option<CameraPoint>;
}

impl<T> TriangulatorRelative for T
where
    T: TriangulatorObservations,
{
    #[inline(always)]
    fn triangulate_relative(
        &self,
        pose: CameraToCamera,
        a: UnitVector3<f64>,
        b: UnitVector3<f64>,
    ) -> Option<CameraPoint> {
        use core::iter::once;

        self.triangulate_observations_to_camera(a, once((pose, b)))
    }
}
