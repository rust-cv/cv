#[cfg(feature = "pinhole")]
use crate::pinhole;
use crate::{Bearing, CameraPoint, FeatureMatch, FeatureWorldMatch, WorldPoint};
use derive_more::{AsMut, AsRef, Deref, DerefMut, From, Into};
use nalgebra::{IsometryMatrix3, Matrix3, Point3, Rotation3, Vector3, SVD};
use sample_consensus::Model;

/// This contains a world pose, which is a pose of the world relative to the camera.
/// This maps [`WorldPoint`] into [`CameraPoint`], changing an absolute position into
/// a vector relative to the camera.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct WorldPose(pub IsometryMatrix3<f64>);

impl<P> Model<FeatureWorldMatch<P>> for WorldPose
where
    P: Bearing,
{
    fn residual(&self, data: &FeatureWorldMatch<P>) -> f32 {
        let WorldPose(iso) = *self;
        let FeatureWorldMatch(feature, world) = data;

        let new_bearing = (iso * world.coords).normalize();
        let bearing_vector = feature.bearing();
        (1.0 - bearing_vector.dot(&new_bearing)) as f32
    }
}

impl WorldPose {
    /// Projects the [`WorldPoint`] into camera space as a [`CameraPoint`].
    pub fn transform(&self, WorldPoint(point): WorldPoint) -> CameraPoint {
        let WorldPose(iso) = *self;
        CameraPoint(iso * point)
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
pub struct CameraPose(pub IsometryMatrix3<f64>);

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
pub struct RelativeCameraPose(pub IsometryMatrix3<f64>);

impl RelativeCameraPose {
    /// Create the pose from rotation and translation.
    pub fn from_parts(rotation: Rotation3<f64>, translation: Vector3<f64>) -> Self {
        Self(IsometryMatrix3::from_parts(translation.into(), rotation))
    }

    /// The relative pose transforms the point in camera space from camera `A` to camera `B`.
    pub fn transform(&self, CameraPoint(point): CameraPoint) -> CameraPoint {
        let Self(iso) = *self;
        CameraPoint(iso * point)
    }

    /// Generates an essential matrix corresponding to this relative camera pose.
    ///
    /// If a point `a` is transformed using [`RelativeCameraPose::transform`] into
    /// a point `b`, then the essential matrix returned by this method will
    /// give a residual of approximately `0.0` when you call
    /// `essential.residual(&FeatureMatch(a, b))`.
    ///
    /// See the documentation of [`EssentialMatrix`] for more information.
    #[cfg_attr(
        feature = "pinhole",
        doc = r##"
        ```
        # use cv_core::{RelativeCameraPose, CameraPoint, FeatureMatch};
        # use cv_core::sample_consensus::Model;
        # use cv_core::nalgebra::{Vector3, Point3, IsometryMatrix3, Rotation3};
        let pose = RelativeCameraPose(IsometryMatrix3::from_parts(
            Vector3::new(0.3, 0.4, 0.5).into(),
            Rotation3::from_euler_angles(0.2, 0.3, 0.4),
        ));
        let a = CameraPoint(Point3::new(0.5, 0.5, 3.0));
        let b = pose.transform(a);
        assert!(pose.essential_matrix().residual(&FeatureMatch(a.into(), b.into())) < 1e-6);
        ```
"##
    )]
    pub fn essential_matrix(&self) -> EssentialMatrix {
        EssentialMatrix(self.0.translation.vector.cross_matrix() * *self.0.rotation.matrix())
    }
}

/// This stores a [`RelativeCameraPose`] that has not had its translation scaled.
///
/// The translation for an unscaled relative camera pose should allow the
/// triangulation of correspondences to lie in front of both cameras.
/// Aside from that case, the relative pose contained inside should only
/// be used to initialize a reconstruction with unknown scale.
#[derive(Debug, Clone, Copy, PartialEq, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct UnscaledRelativeCameraPose(pub RelativeCameraPose);

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
/// plane pointing towards the second camera is called an epipole. The line through the image
/// created by the projected points is called an epipolar line, and it extends from the epipole.
///
/// If you look at every point along a projection out of the camera, each one of those points would
/// project onto the epipolar line on the camera sensor of another image.
/// If you traced every point along the projection to where it would appear on the sensor
/// of the camera (projection of the 3d points into normalized image coordinates), then
/// the points would form the epipolar line. This means that you can draw epipolar lines
/// so long as the projection does not pass through the optical center of both cameras.
/// However, that situation is usually impossible, as one camera would be obscuring the feature
/// for the other camera.
///
/// The essential matrix makes it possible to create a vector that is perpendicular to all
/// bearings that are formed from the epipolar line on the second image's sensor. This is
/// done by computing `E * x`, where `x` is a homogeneous normalized image coordinate
/// from the first image. The transpose of the resulting vector then has a dot product
/// with the transpose of the second image coordinate `x'` which is equal to `0.0`.
/// This can be written as:
///
/// ```text
/// dot(transpose(E * x), x') = 0
/// ```
///
/// This can be re-written into the form given above:
///
/// ```text
/// transpose(x') * E * x = 0
/// ```
///
/// Where the first operation creates a pependicular vector to the epipoles on the first image
/// and the second takes the dot product which should result in 0.
///
/// With a `EssentialMatrix`, you can retrieve the rotation and translation given
/// one normalized image coordinate and one bearing that is scaled to the depth
/// of the point relative to the current reconstruction. This kind of point can be computed
/// using [`WorldPose::transform`] to convert a [`WorldPoint`] to a [`CameraPoint`].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, AsMut, AsRef, Deref, DerefMut, From, Into)]
pub struct EssentialMatrix(pub Matrix3<f64>);

#[cfg(feature = "pinhole")]
impl Model<FeatureMatch<pinhole::NormalizedKeyPoint>> for EssentialMatrix {
    fn residual(&self, data: &FeatureMatch<pinhole::NormalizedKeyPoint>) -> f32 {
        let Self(mat) = *self;
        let FeatureMatch(pinhole::NormalizedKeyPoint(a), pinhole::NormalizedKeyPoint(b)) = *data;

        // The result is a 1x1 matrix which we must get element 0 from.
        libm::fabsf((b.to_homogeneous().transpose() * mat * a.to_homogeneous())[0] as f32)
    }
}

impl EssentialMatrix {
    /// Can be used to enforce the constraints of an essential matrix to fix it.
    ///
    /// This finds the closest essential matrix in frobenius form. This just means
    /// that the two singular values are averaged and the null singular value is
    /// forced to zero.
    pub fn recondition(self, epsilon: f64, max_iterations: usize) -> Option<Self> {
        let old_svd = self.try_svd(true, true, epsilon, max_iterations)?;
        // We need to sort the singular values in the SVD.
        let mut sources = [0, 1, 2];
        sources.sort_unstable_by_key(|&ix| float_ord::FloatOrd(-old_svd.singular_values[ix]));
        let mut svd = old_svd;
        for (dest, &source) in sources.iter().enumerate() {
            svd.singular_values[dest] = old_svd.singular_values[source];
            svd.u
                .as_mut()
                .unwrap()
                .column_mut(dest)
                .copy_from(&old_svd.u.as_ref().unwrap().column(source));
            svd.v_t
                .as_mut()
                .unwrap()
                .row_mut(dest)
                .copy_from(&old_svd.v_t.as_ref().unwrap().row(source));
        }
        // Now that the singular values are sorted, find the closest
        // essential matrix to E in frobenius form.
        // This consists of averaging the two non-zero singular values
        // and zeroing out the near-zero singular value.
        svd.singular_values[2] = 0.0;
        let new_singular = (svd.singular_values[0] + svd.singular_values[1]) / 2.0;
        svd.singular_values[0] = new_singular;
        svd.singular_values[1] = new_singular;
        // Cannot fail because we asked for both U and V* on decomp.
        let mat = svd.recompose().unwrap();
        Some(Self(mat))
    }

    /// Returns two possible rotations for the essential matrix along with a translation
    /// bearing of arbitrary length. The translation bearing is not yet in the correct
    /// space and the inverse rotation (transpose) must be multiplied by the translation
    /// bearing to make the translation bearing be post-rotation. The translation's length
    /// is unknown and of unknown sign and must be solved for by using a prior.
    ///
    /// `epsilon` is the threshold by which the singular value decomposition is considered
    /// complete. Making this smaller may improve the precision. It is recommended to
    /// set this to no higher than `1e-6`.
    ///
    /// `max_iterations` is the maximum number of iterations that singular value decomposition
    /// will run on this matrix. Use this in soft realtime systems to cap the execution time.
    /// A `max_iterations` of `0` may execute indefinitely and is not recommended.
    ///
    /// ```
    /// # use cv_core::RelativeCameraPose;
    /// # use cv_core::nalgebra::{IsometryMatrix3, Rotation3, Vector3};
    /// let pose = RelativeCameraPose(IsometryMatrix3::from_parts(
    ///     Vector3::new(-0.8, 0.4, 0.5).into(),
    ///     Rotation3::from_euler_angles(0.2, 0.3, 0.4),
    /// ));
    /// // Get the possible poses for the essential matrix created from `pose`.
    /// let (rot_a, rot_b, t) = pose.essential_matrix().possible_rotations_unscaled_translation(1e-6, 50).unwrap();
    /// // Compute residual rotations.
    /// let a_res = rot_a.rotation_to(&pose.rotation).angle();
    /// let b_res = rot_b.rotation_to(&pose.rotation).angle();
    /// let a_close = a_res < 1e-4;
    /// let b_close = b_res < 1e-4;
    /// // At least one rotation is correct.
    /// assert!(a_close || b_close);
    /// // The translation points in the same (or reverse) direction
    /// let t_res = 1.0 - t.normalize().dot(&pose.translation.vector.normalize()).abs();
    /// assert!(t_res < 1e-4);
    /// ```
    pub fn possible_rotations_unscaled_translation(
        &self,
        epsilon: f64,
        max_iterations: usize,
    ) -> Option<(Rotation3<f64>, Rotation3<f64>, Vector3<f64>)> {
        let Self(essential) = *self;
        let essential = essential;

        // `W` from https://en.wikipedia.org/wiki/Essential_matrix#Finding_one_solution.
        let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        // Transpose of `W` from https://en.wikipedia.org/wiki/Essential_matrix#Finding_one_solution.
        let wt = w.transpose();

        // Perform SVD.
        let svd = SVD::try_new(essential, true, true, epsilon, max_iterations);
        // Extract only the U and V matrix from the SVD.
        let u_v_t = svd.map(|svd| {
            if let SVD {
                u: Some(u),
                v_t: Some(v_t),
                singular_values,
            } = svd
            {
                // Sort the singular vectors in U and V*.
                let mut sources: [usize; 3] = [0, 1, 2];
                sources.sort_unstable_by_key(|&ix| float_ord::FloatOrd(-singular_values[ix]));
                let mut sorted_u = Matrix3::zeros();
                let mut sorted_v_t = Matrix3::zeros();
                for (&ix, mut column) in sources.iter().zip(sorted_u.column_iter_mut()) {
                    column.copy_from(&u.column(ix));
                }
                for (&ix, mut row) in sources.iter().zip(sorted_v_t.row_iter_mut()) {
                    row.copy_from(&v_t.row(ix));
                }
                (sorted_u, sorted_v_t)
            } else {
                panic!("Didn't get U and V matrix in SVD");
            }
        });
        // Force the determinants to be positive. This is done to ensure the
        // handedness of the rotation matrix is correct.
        let u_v_t = u_v_t.map(|(mut u, mut v_t)| {
            // Last column of U is undetermined since d = (a a 0).
            if u.determinant() < 0.0 {
                for n in u.column_mut(2).iter_mut() {
                    *n *= -1.0;
                }
            }
            // Last row of Vt is undetermined since d = (a a 0).
            if v_t.determinant() < 0.0 {
                for n in v_t.row_mut(2).iter_mut() {
                    *n *= -1.0;
                }
            }
            // Return positive determinant U and V*.
            (u, v_t)
        });
        // Compute the possible rotations and the bearing with no normalization.
        u_v_t.map(|(u, v_t)| {
            (
                Rotation3::from_matrix_unchecked(u * w * v_t),
                Rotation3::from_matrix_unchecked(u * wt * v_t),
                u.column(2).into_owned(),
            )
        })
    }

    /// See [`EssentialMatrix::possible_rotations_unscaled_translation`].
    ///
    /// This returns only the two rotations that are possible.
    ///
    /// ```
    /// # use cv_core::RelativeCameraPose;
    /// # use cv_core::nalgebra::{IsometryMatrix3, Rotation3, Vector3};
    /// let pose = RelativeCameraPose(IsometryMatrix3::from_parts(
    ///     Vector3::new(-0.8, 0.4, 0.5).into(),
    ///     Rotation3::from_euler_angles(0.2, 0.3, 0.4),
    /// ));
    /// // Get the possible rotations for the essential matrix created from `pose`.
    /// let rbs = pose.essential_matrix().possible_rotations(1e-6, 50).unwrap();
    /// let one_correct = rbs.iter().any(|&rot| {
    ///     let angle_residual = rot.rotation_to(&pose.rotation).angle();
    ///     angle_residual < 1e-4
    /// });
    /// assert!(one_correct);
    /// ```
    pub fn possible_rotations(
        &self,
        epsilon: f64,
        max_iterations: usize,
    ) -> Option<[Rotation3<f64>; 2]> {
        self.possible_rotations_unscaled_translation(epsilon, max_iterations)
            .map(|(rot_a, rot_b, _)| [rot_a, rot_b])
    }

    /// See [`EssentialMatrix::possible_rotations_unscaled_translation`].
    ///
    /// This returns the rotations and their corresponding post-rotation translation bearing.
    ///
    /// ```
    /// # use cv_core::RelativeCameraPose;
    /// # use cv_core::nalgebra::{IsometryMatrix3, Rotation3, Vector3};
    /// let pose = RelativeCameraPose(IsometryMatrix3::from_parts(
    ///     Vector3::new(-0.8, 0.4, 0.5).into(),
    ///     Rotation3::from_euler_angles(0.2, 0.3, 0.4),
    /// ));
    /// // Get the possible poses for the essential matrix created from `pose`.
    /// let rbs = pose.essential_matrix().possible_unscaled_poses(1e-6, 50).unwrap();
    /// let one_correct = rbs.iter().any(|&upose| {
    ///     let angle_residual =
    ///         upose.rotation.rotation_to(&pose.rotation).angle();
    ///     let translation_residual =
    ///         1.0 - upose.translation.vector.normalize()
    ///                    .dot(&pose.translation.vector.normalize());
    ///     angle_residual < 1e-4 && translation_residual < 1e-4
    /// });
    /// assert!(one_correct);
    /// ```
    pub fn possible_unscaled_poses(
        &self,
        epsilon: f64,
        max_iterations: usize,
    ) -> Option<[UnscaledRelativeCameraPose; 4]> {
        self.possible_rotations_unscaled_translation(epsilon, max_iterations)
            .map(|(rot_a, rot_b, t)| {
                [
                    UnscaledRelativeCameraPose(RelativeCameraPose::from_parts(rot_a, t)),
                    UnscaledRelativeCameraPose(RelativeCameraPose::from_parts(rot_b, t)),
                    UnscaledRelativeCameraPose(RelativeCameraPose::from_parts(rot_a, -t)),
                    UnscaledRelativeCameraPose(RelativeCameraPose::from_parts(rot_b, -t)),
                ]
            })
    }

    /// Same as [`EssentialMatrix::possible_unscaled_poses`], but it doesn't return
    /// 4 unscaled poses since it doesn't bother to give back the different translation
    /// directions and instead only gives one. This is useful if your algorithm doesn't
    /// care about the direction of translation.
    pub fn possible_unscaled_poses_bearing(
        &self,
        epsilon: f64,
        max_iterations: usize,
    ) -> Option<[UnscaledRelativeCameraPose; 2]> {
        self.possible_rotations_unscaled_translation(epsilon, max_iterations)
            .map(|(rot_a, rot_b, t)| {
                [
                    UnscaledRelativeCameraPose(RelativeCameraPose::from_parts(rot_a, t)),
                    UnscaledRelativeCameraPose(RelativeCameraPose::from_parts(rot_b, t)),
                ]
            })
    }

    /// Finds the unscaled pose that agrees with the most matches.
    pub fn solve_unscaled_pose<P>(
        &self,
        epsilon: f64,
        max_iterations: usize,
        consensus_ratio: f64,
        triangulation_method: impl Fn(UnscaledRelativeCameraPose, P, P) -> Option<Point3<f64>>,
        correspondences: impl Iterator<Item = FeatureMatch<P>>,
    ) -> Option<UnscaledRelativeCameraPose>
    where
        P: Bearing + Copy,
    {
        // Get the possible rotations and the translation
        self.possible_unscaled_poses(epsilon, max_iterations)
            .and_then(|poses| {
                // Get the net translation scale of points that agree with a and b
                // in addition to the number of points that agree with a and b.
                let (ts, total) = correspondences.fold(
                    ([0usize; 4], 0usize),
                    |(mut ts, total), FeatureMatch(a, b)| {
                        let trans_and_agree = |pose| {
                            triangulation_method(pose, a, b)
                                .map(|p| {
                                    let tp = pose.transform(CameraPoint(p));
                                    p.coords.normalize().dot(&a.bearing()) > 0.0
                                        && tp.coords.normalize().dot(&b.bearing()) > 0.0
                                })
                                .unwrap_or(false)
                        };

                        // Do it for all poses.
                        for (tn, &pose) in ts.iter_mut().zip(&poses) {
                            if trans_and_agree(pose) {
                                *tn += 1;
                            }
                        }

                        (ts, total + 1)
                    },
                );

                // Ensure that there is at least one point.
                if total == 0 {
                    return None;
                }

                // Ensure that the best one exceeds the consensus ratio.
                let (ix, best) = ts
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by_key(|&(_, t)| t)
                    .unwrap();
                if (best as f64) < consensus_ratio * total as f64 && best != 0 {
                    return None;
                }

                Some(poses[ix])
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
    /// ```text
    /// transpose(homogeneous(a)) * E * homogeneous(b) = 0
    /// ```
    ///
    /// Also, `a` and `b` must be a correspondence.
    ///
    /// This will take the average translation over the entire iterator. This is done
    /// to smooth out noise and outliers (if present).
    ///
    /// `epsilon` is a small value to which SVD must converge to before terminating.
    ///
    /// `max_iterations` is the maximum number of iterations that SVD will run on this
    /// matrix. Use this to cap the execution time.
    /// A `max_iterations` of `0` may execute indefinitely and is not recommended except
    /// for non-production code.
    ///
    /// `consensus_ratio` is the ratio of points which must be in front of the camera for the model
    /// to be accepted and return Some. Otherwise, None is returned. Set this to about
    /// `0.45` to have approximate majority consensus.
    ///
    /// `bearing_scale` is a function that is passed a translation bearing vector,
    /// an untranslated (but rotated) camera point, and a normalized key point
    /// where the actual point exists. It must return the scalar which the
    /// translation bearing vector must by multiplied by to get the actual translation.
    /// It may return `None` if it fails.
    ///
    /// `correspondences` must provide an iterator of tuples containing the matches
    /// of a 3d `CameraPoint` `a` from camera A and the matching `NormalizedKeyPoint`
    /// `b` from camera B.
    ///
    /// This does not communicate which points were outliers to each model.
    #[cfg_attr(
        feature = "pinhole",
        doc = r##"
        ```
        # use cv_core::{RelativeCameraPose, CameraPoint};
        # use cv_core::nalgebra::{IsometryMatrix3, Rotation3, Vector3, Point3};
        # use cv_core::pinhole::NormalizedKeyPoint;
        let pose = RelativeCameraPose(IsometryMatrix3::from_parts(
            Vector3::new(-0.8, 0.4, 0.5).into(),
            Rotation3::from_euler_angles(0.2, 0.3, 0.4),
        ));
        let a_point = CameraPoint(Point3::new(-1.0, 2.0, 3.0));
        let b_point = pose.transform(a_point);
        // Get the possible poses for the essential matrix created from `pose`.
        let estimate_pose = pose.essential_matrix()
            .solve_pose(1e-6, 50, 0.5,
                cv_core::geom::triangulate_bearing_reproject::<NormalizedKeyPoint>,
                std::iter::once((a_point, b_point.into()))
        ).unwrap();

        let angle_residual =
            estimate_pose.rotation.rotation_to(&pose.rotation).angle();
        let translation_residual = (pose.translation.vector - estimate_pose.translation.vector).norm();
        assert!(angle_residual < 1e-4 && translation_residual < 1e-4, "{}, {}, {:?}, {:?}", angle_residual, translation_residual, pose.translation.vector, estimate_pose.translation.vector);
        ```
"##
    )]
    pub fn solve_pose<P>(
        &self,
        epsilon: f64,
        max_iterations: usize,
        consensus_ratio: f64,
        triangulation_method: impl Fn(Vector3<f64>, CameraPoint, P) -> Option<f64>,
        correspondences: impl Iterator<Item = (CameraPoint, P)> + Clone,
    ) -> Option<RelativeCameraPose>
    where
        P: Bearing + Copy,
    {
        // Get the possible rotations and the translation
        self.possible_unscaled_poses_bearing(epsilon, max_iterations)
            .and_then(|poses| {
                // Get the net translation scale of points that agree with a and b
                // in addition to the number of points that agree with a and b.
                let (ts, total) = correspondences.fold(
                    ([(0usize, 0.0); 2], 0usize),
                    |(mut ts, total), (a, b)| {
                        let trans_and_agree = |pose: UnscaledRelativeCameraPose| {
                            let untranslated = pose.rotation * a.0;
                            let t_scale = triangulation_method(
                                pose.translation.vector,
                                CameraPoint(untranslated),
                                b,
                            )?;

                            if (untranslated.coords + t_scale * pose.translation.vector)
                                .dot(&b.bearing())
                                > 0.0
                            {
                                Some(t_scale)
                            } else {
                                None
                            }
                        };

                        // Do it for all poses.
                        for (tn, &pose) in ts.iter_mut().zip(&poses) {
                            if let Some(scale) = trans_and_agree(pose) {
                                tn.0 += 1;
                                tn.1 += scale;
                            }
                        }

                        (ts, total + 1)
                    },
                );

                // Ensure that there is at least one point.
                if total == 0 {
                    return None;
                }

                // Ensure that the best one exceeds the consensus ratio.
                let (ix, (best_num, scale_acc)) = ts
                    .iter()
                    .copied()
                    .enumerate()
                    .max_by_key(|&(_, (t, _))| t)
                    .unwrap();
                if (best_num as f64) < consensus_ratio * total as f64 && best_num != 0 {
                    return None;
                }
                let scale = scale_acc / best_num as f64;

                Some(RelativeCameraPose::from_parts(
                    poses[ix].rotation,
                    scale * poses[ix].translation.vector,
                ))
            })
    }
}
