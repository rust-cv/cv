use crate::{
    geom, CameraPoint, KeyPointWorldMatch, KeyPointsMatch, NormalizedKeyPoint, WorldPoint,
};
use approx::AbsDiffEq;
use core::cmp::Ordering;
use derive_more::{AsMut, AsRef, Deref, DerefMut, From, Into};
use nalgebra::{
    dimension::{U2, U3, U7},
    Isometry3, Matrix3, Matrix3x2, MatrixMN, Quaternion, Rotation3, Translation3, UnitQuaternion,
    Vector2, Vector3, Vector4, VectorN, SVD,
};
use sample_consensus::Model;

/// This contains a world pose, which is a pose of the world relative to the camera.
/// This maps [`WorldPoint`] into [`CameraPoint`], changing an absolute position into
/// a vector relative to the camera.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct WorldPose(pub Isometry3<f32>);

impl Model<KeyPointWorldMatch> for WorldPose {
    fn residual(&self, data: &KeyPointWorldMatch) -> f32 {
        let WorldPose(iso) = *self;
        let KeyPointWorldMatch(image, world) = *data;

        let new_bearing = (iso * world.coords).normalize();
        let bearing_vector = image.to_homogeneous().normalize();
        1.0 - bearing_vector.dot(&new_bearing)
    }
}

impl WorldPose {
    /// Computes difference between the image keypoint and the projected keypoint.
    pub fn projection_error(&self, correspondence: KeyPointWorldMatch) -> Vector2<f32> {
        let KeyPointWorldMatch(NormalizedKeyPoint(image), world) = correspondence;
        let NormalizedKeyPoint(projection) = self.project(world);
        image - projection
    }

    /// Projects the `WorldPoint` onto the camera as a `NormalizedKeyPoint`.
    pub fn project(&self, point: WorldPoint) -> NormalizedKeyPoint {
        self.transform(point).into()
    }

    /// Projects the [`WorldPoint`] into camera space as a [`CameraPoint`].
    pub fn transform(&self, WorldPoint(point): WorldPoint) -> CameraPoint {
        let WorldPose(iso) = *self;
        CameraPoint((iso * point).coords)
    }

    /// Computes the Jacobian of the projection in respect to the `WorldPose`.
    /// The Jacobian is in the format:
    /// ```no_build,no_run
    /// | dx/dtx dy/dPx |
    /// | dx/dty dy/dPy |
    /// | dx/dtz dy/dPz |
    /// | dx/dqr dy/dqr |
    /// | dx/dqx dy/dqx |
    /// | dx/dqy dy/dqy |
    /// | dx/dqz dy/dqz |
    /// ```
    ///
    /// Where `t` refers to the translation vector and `q` refers to the unit quaternion.
    #[rustfmt::skip]
    pub fn projection_pose_jacobian(&self, point: WorldPoint) -> MatrixMN<f32, U7, U2> {
        let q = self.0.rotation.quaternion().coords;
        // World point (input)
        let p = point.0.coords;
        // Camera point (intermediate output)
        let pc = (self.0 * point.0).coords;

        // dP/dT (Jacobian of camera point in respect to translation component)
        let dp_dt = Matrix3::identity();

        // d/dQv (Qv x (Qv x P))
        let qv_qv_p = Matrix3::new(
            q.y * p.y + q.z * p.z,          q.y * p.x - 2.0 * q.x * p.y,    q.z * p.x - 2.0 * q.x * p.z,
            q.x * p.y - 2.0 * q.y * p.x,    q.x * p.x + q.z * p.z,          q.z * p.y - 2.0 * q.y * p.z,
            q.x * p.z - 2.0 * q.z * p.x,    q.y * p.z - 2.0 * q.z * p.y,    q.x * p.x + q.y * p.y
        );
        // d/dQv (Qv x P)
        let qv_p = Matrix3::new(
            0.0,    -p.z,   p.y,
            p.z,    0.0,    -p.x,
            -p.y,   p.x,    0.0,
        );
        // dP/dQv = d/dQv (2 * Qs * Qv x P + 2 * Qv x (Qv x P))
        // Jacobian of camera point in respect to vector component of quaternion
        let dp_dqv = 2.0 * (q.w * qv_p + qv_qv_p);

        // dP/Ds = d/Qs (2 * Qs * Qv x P)
        // Jacobian of camera point in respect to real component of quaternion
        let dp_ds = 2.0 * q.xyz().cross(&p);

        // dP/dT,Q (Jacobian of 3d camera point in respect to translation and quaternion)
        let dp_dtq = MatrixMN::<f32, U7, U3>::from_rows(&[
            dp_dt.row(0).into(),
            dp_dt.row(1).into(),
            dp_dt.row(2).into(),
            dp_dqv.row(0).into(),
            dp_dqv.row(1).into(),
            dp_dqv.row(2).into(),
            dp_ds.transpose(),
        ]);

        // 1 / pz
        let pcz = pc.z.recip();
        // - 1 / pz^2
        let npcz2 = -(pcz * pcz);

        // dK/dp (Jacobian of normalized image coordinate in respect to 3d camera point)
        let dk_dp = Matrix3x2::new(
            pcz,    0.0,
            0.0,    pcz,
            npcz2,  npcz2,
        );

        dp_dtq * dk_dp
    }

    pub fn to_vec(&self) -> VectorN<f32, U7> {
        let Self(iso) = *self;
        let t = iso.translation.vector;
        let rc = iso.rotation.quaternion().coords;
        t.push(rc.x).push(rc.y).push(rc.z).push(rc.w)
    }

    pub fn from_vec(v: VectorN<f32, U7>) -> Self {
        Self(Isometry3::from_parts(
            Translation3::from(Vector3::new(v[0], v[1], v[2])),
            UnitQuaternion::from_quaternion(Quaternion::from(Vector4::new(v[3], v[4], v[5], v[6]))),
        ))
    }
}

impl From<CameraPose> for WorldPose {
    fn from(camera: CameraPose) -> Self {
        Self(camera.inverse())
    }
}

/// This contains a camera pose, which is a pose of the camera relative to the world.
/// This transforms camera points (with depth as `z`) into world coordinates.
/// This also tells you where the camera is located and oriented in the world.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct CameraPose(pub Isometry3<f32>);

impl From<WorldPose> for CameraPose {
    fn from(world: WorldPose) -> Self {
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
pub struct RelativeCameraPose(pub Isometry3<f32>);

impl RelativeCameraPose {
    // The relative pose transforms the point in camera space from camera `A` to camera `B`.
    pub fn transform(&self, CameraPoint(point): CameraPoint) -> CameraPoint {
        let Self(iso) = *self;
        CameraPoint(iso * point)
    }
}

/// This stores an essential matrix, which is satisfied by the following constraint:
///
/// transpose(x') * E * x = 0
///
/// Where `x'` and `x` are homogeneous normalized image coordinates. You can get a
/// homogeneous normalized image coordinate by appending `1.0` to a `NormalizedKeyPoint`.
///
/// The essential matrix embodies the epipolar constraint between two images. Given that light
/// travels in a perfectly straight line (it will not, but for short distances it mostly does)
/// and assuming a pinhole camera model, for any point on the camera sensor, the light source
/// for that point exists somewhere along a line extending out from the bearing (direction
/// of travel) of that point. For a normalized image coordinate, that bearing is `(x, y, 1.0)`.
/// That is because normalized image coordinates exist on a virtual plane (the sensor)
/// a distance `z = 1.0` from the optical center (the location of the focal point) where
/// the unit of distance is the focal length. In epipolar geometry, the point on the virtual
/// plane is called an epipole. The line through 3d space created by the bearing that travels
/// from the optical center through the epipole is called an epipolar line.
///
/// If you look at every point along an epipolar line, each one of those points would show
/// up as a different point on the camera sensor of another image (if they are in view).
/// If you traced every point along this epipolar line to where it would appear on the sensor
/// of the camera (projection of the 3d points into normalized image coordinates), then
/// the points would form a straight line. This means that you can draw epipolar lines
/// that do not pass through the optical center of an image on that image.
///
/// The essential matrix makes it possible to create a vector that is perpendicular to all
/// bearings that are formed from the epipolar line on the second image's sensor. This is
/// done by computing `E * x`, where `x` is a homogeneous normalized image coordinate
/// from the first image. The transpose of the resulting vector then has a dot product
/// with the transpose of the second image coordinate `x'` which is equal to `0.0`.
/// This can be written as:
///
/// ```no_build,no_run
/// dot(transpose(E * x), x') = 0
/// ```
///
/// This can be re-written into the form given above:
///
/// ```no_build,no_run
/// transpose(x') * E * x = 0
/// ```
///
/// Where the first operation creates a pependicular vector to the epipoles on the first image
/// and the second takes the dot product which should result in 0.
///
/// With a `EssentialMatrix`, you can retrieve the rotation and translation given
/// one normalized image coordinate and one bearing that is scaled to the depth
/// of the point relative to the current reconstruction. This kind of point can be computed
/// using [`WorldPose::project_camera`] to convert a [`WorldPoint`] to a [`CameraPoint`].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct EssentialMatrix(pub Matrix3<f32>);

impl Model<KeyPointsMatch> for EssentialMatrix {
    fn residual(&self, data: &KeyPointsMatch) -> f32 {
        let Self(mat) = *self;
        let KeyPointsMatch(NormalizedKeyPoint(a), NormalizedKeyPoint(b)) = *data;

        // The result is a 1x1 matrix which we must get element 0 from.
        (a.to_homogeneous().transpose() * mat * b.to_homogeneous())[0]
    }
}

impl EssentialMatrix {
    /// Returns two possible rotations for the essential matrix along with a translation
    /// direction of unit length. The translation's length is unknown and must be
    /// solved for by using a prior.
    ///
    /// `max_iterations` is the maximum number of iterations that singular value decomposition
    /// will run on this matrix. Use this in soft realtime systems to cap the execution time.
    /// A `max_iterations` of `0` may execute indefinitely and is not recommended.
    pub fn possible_poses(
        &self,
        max_iterations: usize,
    ) -> Option<(Rotation3<f32>, Rotation3<f32>, Vector3<f32>)> {
        let Self(essential) = *self;

        // `W` from https://en.wikipedia.org/wiki/Essential_matrix#Finding_one_solution.
        let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        // Transpose of `W` from https://en.wikipedia.org/wiki/Essential_matrix#Finding_one_solution.
        let wt = w.transpose();

        // Perform SVD.
        let svd = SVD::try_new(
            essential,
            true,
            true,
            f32::default_epsilon(),
            max_iterations,
        );
        // Extract only the U and V matrix from the SVD.
        let u_v = svd.map(|svd| {
            if let SVD {
                u: Some(u),
                v_t: Some(v_t),
                ..
            } = svd
            {
                (u, v_t.transpose())
            } else {
                panic!("Didn't get U and V matrix in SVD");
            }
        });
        // Force the determinants to be positive. I do not know precisely
        // why this is done since it isn't apparent from the Wikipedia page
        // on this subject, but this is what TheiaSfM does in essential_matrix_utils.cc.
        let u_v = u_v.map(|(mut u, mut v)| {
            let fix_det = |m: &mut Matrix3<f32>| {
                if m.determinant() < 0.0 {
                    for n in m.column_mut(2).iter_mut() {
                        *n *= -1.0;
                    }
                }
            };
            fix_det(&mut u);
            fix_det(&mut v);
            (u, v)
        });
        // Compute the possible rotations and the normalized bearing.
        u_v.map(|(u, v)| {
            let vt = v.transpose();
            (
                Rotation3::from_matrix_unchecked(u * wt * vt),
                Rotation3::from_matrix_unchecked(u * w * vt),
                u.column(2).into_owned().normalize(),
            )
        })
    }

    /// Return the [`RelativeCameraPose`] that transforms a [`CameraPoint`] of image
    /// `A` (source of `a`) to the corresponding [`CameraPoint`] of image B (source of `b`).
    /// This determines the average expected translation from the points themselves and
    /// if the points agree with the rotation (points must be in front of the camera).
    /// The function takes an iterator containing tuples in the form `(depth, a, b)`:
    ///
    /// * `depth` - The actual depth (`z` axis, not distance) of normalized keypoint `a`
    /// * `a` - A keypoint from image `A`
    /// * `b` - A keypoint from image `B`
    ///
    /// `self` must satisfy the constraint:
    ///
    /// ```no_build,no_run
    /// transpose(homogeneous(a)) * E * homogeneous(b) = 0
    /// ```
    ///
    /// Also, `a` and `b` must be a correspondence.
    ///
    /// This will take the average translation over the entire iterator. This is done
    /// to smooth out noise and outliers (if present).
    ///
    /// `consensus_ratio` is the ratio of points which must be in front of the camera for the model
    /// to be accepted and return Some. Otherwise, None is returned.
    ///
    /// `max_iterations` is the maximum number of iterations that singular value decomposition
    /// will run on this matrix. Use this in soft realtime systems to cap the execution time.
    /// A `max_iterations` of `0` may execute indefinitely and is not recommended.
    ///
    /// This does not communicate which points were outliers. To determine the outlier points,
    /// get the [`CameraPoint`] for all points and place them in front of
    pub fn solve_pose(
        &self,
        consensus_ratio: f32,
        max_iterations: usize,
        correspondences: impl Iterator<Item = (f32, NormalizedKeyPoint, NormalizedKeyPoint)>,
    ) -> Option<RelativeCameraPose> {
        // Get the possible rotations and the translation
        self.possible_poses(max_iterations)
            .and_then(|(rot_x, rot_y, t)| {
                // Find the translation for both x and y.
                let tx_dir = rot_x.transpose() * t;
                let ty_dir = rot_y.transpose() * t;
                // Get the net translation scale of points that agree with a and b
                // in addition to the number of points that agree with a and b.
                let (xt, yt, xn, yn, total) = correspondences.fold(
                    (0.0, 0.0, 0usize, 0usize, 0usize),
                    |(mut xt, mut yt, mut xn, mut yn, total), (depth, a, b)| {
                        // Compute the CameraPoint of a.
                        let a_point = a.with_depth(depth).0;
                        let trans_and_agree = |rotation, t_dir| {
                            // Triangulate the position of the CameraPoint of b.
                            // We know the precise 3d position of the a point relative
                            // to camera A, but we do not know the
                            // 3d point in relation to camera B since the translation of
                            // the point is unknown. We do know the direction of translation
                            // of the point. We know only the rotation of the camera B
                            // relative to camera A and the epipolar point on camera B.
                            // What we will need to do is start by rotating the point in space.
                            // After rotating the point, we then need to solve for the translation
                            // that minimizes the reprojection error of the untranslated point as much
                            // as possible. See the documentation for reproject_along_translation
                            // to get more details on the process.
                            let untranslated: Vector3<f32> = rotation * a_point;
                            let translation_scale =
                                geom::reproject_along_translation(untranslated.xy(), b, t_dir);
                            // Now that we have the translation, we can just verify that the point
                            // is in front (z > 1.0) of the camera to see if it agrees with the model.
                            (
                                translation_scale,
                                translation_scale * t_dir.z + untranslated.z > 1.0,
                            )
                        };

                        // Do it for X.
                        if let (scale, true) = trans_and_agree(rot_x, tx_dir) {
                            xt += scale;
                            xn += 1;
                        }

                        // Do it for Y.
                        if let (scale, true) = trans_and_agree(rot_y, ty_dir) {
                            yt += scale;
                            yn += 1;
                        }

                        (xt, yt, xn, yn, total + 1)
                    },
                );
                // Ensure that the best one exceeds the consensus ratio.
                if (core::cmp::max(xn, yn) as f32 / total as f32) < consensus_ratio {
                    return None;
                }
                // TODO: Perhaps if its closer than this we should assume the frame itself is an outlier.
                let (rot, trans) = match xn.cmp(&yn) {
                    Ordering::Equal => return None,
                    Ordering::Greater => (rot_x, xt / xn as f32 * tx_dir),
                    Ordering::Less => (rot_y, yt / yn as f32 * ty_dir),
                };
                Some(RelativeCameraPose(Isometry3::from_parts(
                    trans.into(),
                    UnitQuaternion::from_rotation_matrix(&rot),
                )))
            })
    }
}
