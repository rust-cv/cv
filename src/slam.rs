// use cv_core::nalgebra::Vector3;
// use cv_core::CameraModel;
// use cv_pinhole::CameraIntrinsics;
// use evmap::{ReadHandle, WriteHandle};
// use sharded_slab::Slab;

// pub struct Appearance {
//     feed: usize,
//     image: usize,
// }

// pub struct Landmark {
//     pub position: Vector3<f64>,
//     pub appearances: Slab<Appearance>,
// }

// pub struct Slam {
//     /// Contains the camera intrinsics for each feed
//     feeds: Slab<CameraIntrinsics>,
//     /// Contains all the landmarks
//     landmarks: Slab<Landmark>,
// }
