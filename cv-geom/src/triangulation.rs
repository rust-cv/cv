use crate::epipolar;
use cv_core::{
    nalgebra::{zero, Matrix3x4, Matrix4, RowVector4, UnitVector3, Vector3},
    CameraPoint, CameraToCamera, Pose, Projective, TriangulatorObservations, TriangulatorRelative,
    WorldPoint, WorldToCamera,
};

/// This is a very quick triangulator to execute, but it is not particularly suitable for optimization.
/// It can be used for optimization when you have very low error to begin with.
/// It is suitable for quickly generating 3d point outputs, such as for display purposes.
/// It is not suitable for accurately computing points at infinity in projective space.
/// It is not suitable for camera models with very high FoV.
///
/// Reffered to as the Linear-Eigen method by Hartley and Sturm in the paper
/// ["Triangulation"](https://users.cecs.anu.edu.au/~hartley/Papers/triangulation/triangulation.pdf).
///
/// This method works by observing that each of the components outputted by the transformation of the
/// [`WorldToCamera`] matrix are in `<x, y, z>` in world-space and `<xz, yz>` in image space.
/// The goal is to minimize `||<xz, yz>||``, the squared reprojection error.
/// By substituting the linear equation for `z` into this equation, you get a series of 4 linear
/// equations that minimize the squared reprojection error. A symmetric eigen decomposition is utilized
/// to get the result. Because symmetric eigen decomposition is fast, the matrix is only 4x4,
/// and the method is totally linear, this can scale effortlessly to many points and is incredibly fast to
/// solve. However, it has serious drawbacks in terms of accuracy.
///
/// ```
/// use cv_core::nalgebra::{Vector3, Point3, Rotation3};
/// use cv_core::{TriangulatorRelative, CameraToCamera, CameraPoint, Pose, Projective};
/// use cv_geom::triangulation::LinearEigenTriangulator;
///
/// let point = CameraPoint::from_point(Point3::new(0.3, 0.1, 2.0));
/// let pose = CameraToCamera::from_parts(Vector3::new(0.1, 0.1, 0.1), Rotation3::new(Vector3::new(0.1, 0.1, 0.1)));
/// let bearing_a = point.bearing();
/// let bearing_b = pose.transform(point).bearing();
/// let triangulated = LinearEigenTriangulator::new().triangulate_relative(pose, bearing_a, bearing_b).unwrap();
/// let distance = (point.point().unwrap().coords - triangulated.point().unwrap().coords).norm();
/// assert!(distance < 1e-6);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct LinearEigenTriangulator {
    epsilon: f64,
    max_iterations: usize,
}

impl LinearEigenTriangulator {
    /// Creates a `LinearEigenTriangulator` with default values.
    ///
    /// Same as calling [`Default::default`].
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the epsilon used in the symmetric eigen solver.
    ///
    /// Default is `1e-12`.
    #[must_use]
    pub fn epsilon(self, epsilon: f64) -> Self {
        Self { epsilon, ..self }
    }

    /// Set the maximum number of iterations for the symmetric eigen solver.
    ///
    /// Default is `1000`.
    #[must_use]
    pub fn max_iterations(self, max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..self
        }
    }
}

impl Default for LinearEigenTriangulator {
    fn default() -> Self {
        Self {
            epsilon: 1e-12,
            max_iterations: 1000,
        }
    }
}

impl TriangulatorObservations for LinearEigenTriangulator {
    fn triangulate_observations(
        &self,
        mut pairs: impl Iterator<Item = (WorldToCamera, UnitVector3<f64>)> + Clone,
    ) -> Option<WorldPoint> {
        if pairs.clone().count() < 2 {
            return None;
        }

        let mut a: Matrix4<f64> = zero();
        for (pose, bearing) in pairs.clone() {
            let bearing = bearing.into_inner();
            // Get the pose as a 3x4 matrix.
            let rot = pose.0.rotation.matrix();
            let trans = pose.0.translation.vector;
            let pose = Matrix3x4::<f64>::from_columns(&[
                rot.column(0),
                rot.column(1),
                rot.column(2),
                trans.column(0),
            ]);
            // Set up the least squares problem.
            let term = pose - bearing * bearing.transpose() * pose;
            a += term.transpose() * term;
        }

        let se = a.try_symmetric_eigen(self.epsilon, self.max_iterations)?;

        // Find the smallest eigenvalue where our point will lie in the null space homogeneous vector.
        se.eigenvalues
            .iter()
            .enumerate()
            .min_by_key(|&(_, &n)| float_ord::FloatOrd(n))
            .map(|(ix, _)| se.eigenvectors.column(ix).into_owned())
            .map(WorldPoint::from_homogeneous)
            .filter(|point| {
                // Ensure the point contains no NaN or infinity.
                point.homogeneous().iter().all(|n| n.is_finite())
            })
            .filter(|point| {
                // Ensure the cheirality constraint.
                pairs.all(|(pose, bearing)| {
                    let pose = pose.inverse().isometry();
                    let bearing = pose * bearing;
                    bearing.dot(&point.bearing()).is_sign_positive()
                })
            })
    }
}

/// This is a very quick triangulator to execute, but it is not particularly suitable for optimization.
/// It can be used for optimization when you have very low error to begin with.
/// It is suitable for quickly generating 3d point outputs, such as for display purposes.
/// It is not suitable for accurately computing points at infinity in projective space.
/// It is not suitable for camera models with very high FoV.
///
/// Reffered to as the Linear-Eigen method by Hartley and Sturm in the paper
/// ["Triangulation"](https://users.cecs.anu.edu.au/~hartley/Papers/triangulation/triangulation.pdf).
///
/// This method works by observing that each of the components outputted by the transformation of the
/// [`WorldToCamera`] matrix are in `<x, y, z>` in world-space and `<xz, yz>` in image space.
/// The goal is to minimize `||<xz, yz>||``, the squared reprojection error.
/// By substituting the linear equation for `z` into this equation, you get a series of 4 linear
/// equations that minimize the squared reprojection error. A symmetric eigen decomposition is utilized
/// to get the result. Because symmetric eigen decomposition is fast, the matrix is only 4x4,
/// and the method is totally linear, this can scale effortlessly to many points and is incredibly fast to
/// solve. However, it has serious drawbacks in terms of accuracy.
///
/// ```
/// use cv_core::nalgebra::{Vector3, Point3, Rotation3};
/// use cv_core::{TriangulatorRelative, CameraToCamera, CameraPoint, Pose, Projective};
/// use cv_geom::triangulation::SineL1Triangulator;
///
/// let point = CameraPoint::from_point(Point3::new(0.3, 0.1, 2.0));
/// let pose = CameraToCamera::from_parts(Vector3::new(0.1, 0.1, 0.1), Rotation3::new(Vector3::new(0.1, 0.1, 0.1)));
/// let bearing_a = point.bearing();
/// let bearing_b = pose.transform(point).bearing();
/// let triangulated = SineL1Triangulator::new().triangulate_relative(pose, bearing_a, bearing_b).unwrap();
/// let distance = (point.point().unwrap().coords - triangulated.point().unwrap().coords).norm();
/// assert!(distance < 1e-6, "real: {}\ntriangulated:{}", point.point().unwrap().coords, triangulated.point().unwrap().coords);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct SineL1Triangulator {
    epsilon: f64,
    max_iterations: usize,
    optimization_rate: f64,
}

impl SineL1Triangulator {
    /// Creates a `SineL1Triangulator` with default values.
    ///
    /// Same as calling [`Default::default`].
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the epsilon used in the symmetric eigen solver.
    ///
    /// Default is `1e-12`.
    #[must_use]
    pub fn epsilon(self, epsilon: f64) -> Self {
        Self { epsilon, ..self }
    }

    /// Set the maximum number of iterations for the symmetric eigen solver.
    ///
    /// Default is `1000`.
    #[must_use]
    pub fn max_iterations(self, max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..self
        }
    }

    /// This sets the optimization rate. This is the the speed of gradient descent.
    ///
    /// Default is `0.01`.
    #[must_use]
    pub fn optimization_rate(self, optimization_rate: f64) -> Self {
        Self {
            optimization_rate,
            ..self
        }
    }

    /// The linear eigen triangulator is used internally by this triangulator to create
    /// the initial guess. This generates it.
    fn linear_eigen_triangulator(&self) -> LinearEigenTriangulator {
        LinearEigenTriangulator {
            epsilon: self.epsilon,
            max_iterations: self.max_iterations,
        }
    }
}

impl Default for SineL1Triangulator {
    fn default() -> Self {
        Self {
            epsilon: 1e-12,
            max_iterations: 1000,
            optimization_rate: 1.0,
        }
    }
}

impl TriangulatorObservations for SineL1Triangulator {
    fn triangulate_observations(
        &self,
        pairs: impl Iterator<Item = (WorldToCamera, UnitVector3<f64>)> + Clone,
    ) -> Option<WorldPoint> {
        let epsilon_squared = self.epsilon * self.epsilon;
        // Create the initial guess using a fast method.
        let point = self
            .linear_eigen_triangulator()
            .triangulate_observations(pairs.clone())?;

        // Convert the point to 3d from projective, otherwise the following will not do anything and we can quit now.
        let mut point = if let Some(point) = point.point() {
            point
        } else {
            return Some(point);
        };

        let scale = self.optimization_rate / pairs.clone().count() as f64;

        // Now we need to refine the point repeatedly.
        for _ in 0..self.max_iterations {
            let delta = scale
                * pairs
                    .clone()
                    .map(|(observation_pose, observation_bearing)| {
                        // The pose needs to go from the camera to the world, so we must reverse it.
                        let observation_pose = observation_pose.isometry().inverse();
                        // The bearing needs to be moved into the world space from the camera space.
                        let observation_bearing = observation_pose * observation_bearing;
                        // Find the epipolar gradient that can move the world to make the point's bearing intersect
                        // with the epipolar plane of this observation.
                        epipolar::point_gradient(
                            observation_pose.translation.vector - point.coords,
                            observation_bearing,
                        )
                    })
                    .sum::<Vector3<f64>>();
            point += delta;

            // Quit if we reach the accuracy of the epsilon.
            if delta.norm_squared() / point.coords.norm_squared() < epsilon_squared {
                break;
            }
        }

        Some(Projective::from_point(point))
    }
}

/// Based on algorithm 12 from "Multiple View Geometry in Computer Vision, Second Edition"
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct RelativeDltTriangulator {
    epsilon: f64,
    max_iterations: usize,
}

impl RelativeDltTriangulator {
    /// Creates a `RelativeDltTriangulator` with default values.
    ///
    /// Same as calling [`Default::default`].
    pub fn new() -> Self {
        Default::default()
    }

    /// Set the epsilon used in the SVD solver.
    ///
    /// Default is `1e-9`.
    #[must_use]
    pub fn epsilon(self, epsilon: f64) -> Self {
        Self { epsilon, ..self }
    }

    /// Set the maximum number of iterations for the SVD solver.
    ///
    /// Default is `100`.
    #[must_use]
    pub fn max_iterations(self, max_iterations: usize) -> Self {
        Self {
            max_iterations,
            ..self
        }
    }
}

impl Default for RelativeDltTriangulator {
    fn default() -> Self {
        Self {
            epsilon: 1e-12,
            max_iterations: 1000,
        }
    }
}

impl TriangulatorRelative for RelativeDltTriangulator {
    fn triangulate_relative(
        &self,
        relative_pose: CameraToCamera,
        a: UnitVector3<f64>,
        b: UnitVector3<f64>,
    ) -> Option<CameraPoint> {
        let pose = relative_pose.homogeneous();
        let mut design = Matrix4::zeros();
        design
            .row_mut(0)
            .copy_from(&RowVector4::new(-a.z, 0.0, a.x, 0.0));
        design
            .row_mut(1)
            .copy_from(&RowVector4::new(0.0, -a.z, a.y, 0.0));
        design
            .row_mut(2)
            .copy_from(&(b.x * pose.row(2) - b.z * pose.row(0)));
        design
            .row_mut(3)
            .copy_from(&(b.y * pose.row(2) - b.z * pose.row(1)));

        let svd = design.try_svd(false, true, self.epsilon, self.max_iterations)?;

        // Extract the null-space vector from V* corresponding to the smallest
        // singular value, which is the homogeneous coordiate of the output.
        Some(svd.v_t.unwrap().row(3).transpose())
            .map(CameraPoint::from_homogeneous)
            .filter(|point| {
                // Ensure the point contains no NaN or infinity.
                point.homogeneous().iter().all(|x| x.is_finite())
            })
            .filter(|point| {
                // Ensure the cheirality constraint.
                point.bearing().dot(&a).is_sign_positive()
                    && point
                        .bearing()
                        .dot(&(relative_pose.isometry().inverse() * b))
                        .is_sign_positive()
            })
    }
}

/// I don't recommend using this triangulator.
///
/// This triangulator creates a skew line that starts at the average position of cameras
/// and points down the average bearing. It then finds the average closest point
/// on the average bearing to each other bearing. This point is the point that is returned.
///
/// # Example
/// ```
/// use cv_geom::triangulation::MeanMeanTriangulator;
/// use cv_core::{nalgebra::{Vector3, Rotation3}, CameraToCamera, Pose, Projective, CameraPoint, TriangulatorRelative};
/// // Create a pose.
/// let pose = CameraToCamera::from_parts(Vector3::new(0.1, 0.1, 0.1), Rotation3::from_scaled_axis(Vector3::new(0.1, 0.1, 0.1)));
/// // Create a point in front of both cameras and between both cameras.
/// let real_point = CameraPoint::from_point(Vector3::new(0.3, 0.1, 2.0).into());
/// // Turn the points into bearings in each camera and try to triangulate the point again.
/// let triangulated_point = MeanMeanTriangulator.triangulate_relative(
///     pose,
///     real_point.bearing(),
///     pose.transform(real_point).bearing()
/// ).unwrap().point().unwrap();
/// // Verify that the point is approximately equal.
/// let real_point = real_point.point().unwrap();
/// assert!((real_point - triangulated_point).norm() < 1e-2, "real_point: {}, triangulated_point: {}", real_point, triangulated_point);
/// ```
#[derive(Copy, Clone, Debug, Default)]
pub struct MeanMeanTriangulator;

impl TriangulatorObservations for MeanMeanTriangulator {
    fn triangulate_observations(
        &self,
        mut pairs: impl Iterator<Item = (WorldToCamera, UnitVector3<f64>)> + Clone,
    ) -> Option<WorldPoint> {
        let total = pairs.clone().count() as f64;
        let (sum_center, sum_bearings) = pairs.clone().fold(
            (Vector3::zeros(), Vector3::zeros()),
            |(sum_center, sum_bearings), (pose, bearing)| {
                let pose = pose.inverse().isometry();
                let bearing = pose * bearing;
                let position = pose.translation.vector;
                (sum_center + position, sum_bearings + bearing.into_inner())
            },
        );
        let average_center = sum_center / total;
        let average_bearing = sum_bearings.normalize();

        let average_projection_distance = pairs
            .clone()
            .map(|(pose, bearing)| {
                let pose = pose.inverse().isometry();
                let bearing = pose * bearing;
                let position = pose.translation.vector;
                let trans = average_center - position;

                let q = average_bearing.cross(&bearing);
                q.scale(q.norm_squared().recip())
                    .dot(&(bearing.cross(&trans)))
            })
            .sum::<f64>()
            / total;

        let w = average_projection_distance.recip();
        Some(WorldPoint::from_homogeneous(
            (average_bearing + average_center * w).push(w),
        ))
        .filter(|point| {
            // Ensure the point contains no NaN or infinity.
            point.homogeneous().iter().all(|n| n.is_finite())
        })
        .filter(|point| {
            // Ensure the cheirality constraint.
            pairs.all(|(pose, bearing)| {
                let pose = pose.inverse().isometry();
                let bearing = pose * bearing;
                bearing.dot(&point.bearing()).is_sign_positive()
            })
        })
    }
}

/// let point = CameraPoint::from_point(Point3::new(0.3, 0.1, 2.0));
/// let pose = CameraToCamera::from_parts(Vector3::new(0.1, 0.1, 0.1), Rotation3::new(Vector3::new(0.1, 0.1, 0.1)));
/// From the paper "Closed-Form Optimal Two-View Triangulation Based on Angular Errors"
/// in section 5: "Closed-Form L1 Triangulation".
///
/// It triangulates by minimizing the L1 angular distance
///
/// # Example
/// ```
/// use cv_geom::triangulation::AngularL1Triangulator;
/// use cv_core::{nalgebra::{Vector3, Rotation3}, CameraToCamera, Pose, Projective, CameraPoint, TriangulatorRelative};
/// // Create a pose.
/// let pose = CameraToCamera::from_parts(Vector3::new(0.1, 0.1, 0.1), Rotation3::from_scaled_axis(Vector3::new(0.1, 0.1, 0.1)));
/// // Create a point in front of both cameras and between both cameras.
/// let real_point = CameraPoint::from_point(Vector3::new(0.3, 0.1, 2.0).into());
/// // Turn the points into bearings in each camera and try to triangulate the point again.
/// let triangulated_point = AngularL1Triangulator.triangulate_relative(
///     pose,
///     real_point.bearing(),
///     pose.transform(real_point).bearing()
/// ).unwrap().point().unwrap();
/// // Verify that the point is approximately equal.
/// let real_point = real_point.point().unwrap();
/// assert!((real_point - triangulated_point).norm() < 1e-6, "real_point: {}, triangulated_point: {}", real_point, triangulated_point);
/// ```
#[derive(Copy, Clone, Debug, Default)]
pub struct AngularL1Triangulator;

impl TriangulatorRelative for AngularL1Triangulator {
    fn triangulate_relative(
        &self,
        original_relative_pose: CameraToCamera,
        a: UnitVector3<f64>,
        b: UnitVector3<f64>,
    ) -> Option<CameraPoint> {
        // The algorithm in the paper triangulates the point from the perspective of the second camera,
        // so this line flips and shadows everything to ensure that the triangulation happens in the first camera.
        // Technically, everything could have been reversed below, but it would be tedious and it would
        // not reflect the paper.
        let relative_pose = original_relative_pose.inverse();
        let (a, b) = (b, a);

        // Transform a into the perspective of the second camera.
        let a = relative_pose.isometry() * a;
        let translation = relative_pose.isometry().translation.vector;
        let normalized_translation = translation.normalize();
        // Correct a and b to intersect at the point which minimizes L1 distance as per
        // "Closed-Form Optimal Two-View Triangulation Based on Angular Errors" algorithm
        // 12 and 13.
        let cross_a = a.cross(&normalized_translation);
        let cross_a_norm = cross_a.norm();
        let na = cross_a / cross_a_norm;
        let cross_b = b.cross(&normalized_translation);
        let cross_b_norm = cross_b.norm();
        let nb = cross_b / cross_b_norm;
        // Shadow the old a and b, as they have been corrected.
        let (a, b) = if cross_a_norm < cross_b_norm {
            // Algorithm 12.
            // This effectively computes the sine of the angle between the plane formed between b
            // and translation and the bearing formed by a. It then multiplies this by the normal vector
            // of the plane (nb) to get the normal corrective factor that is applied to a.
            let new_a = UnitVector3::new_normalize(a.into_inner() - (a.dot(&nb) * nb));
            (new_a, b)
        } else {
            // Algorithm 13.
            // This effectively computes the sine of the angle between the plane formed between a
            // and translation and the bearing formed by b. It then multiplies this by the normal vector
            // of the plane (na) to get the normal corrective factor that is applied to b.
            let new_b = UnitVector3::new_normalize(b.into_inner() - (b.dot(&na) * na));
            (a, new_b)
        };

        let z = b.cross(&a);
        Some(CameraPoint::from_homogeneous(
            b.into_inner()
                .push(z.norm_squared() / z.dot(&translation.cross(&a))),
        ))
        .filter(|point| {
            // Ensure the point contains no NaN or infinity.
            point.homogeneous().iter().all(|n| n.is_finite())
        })
        .filter(|point| {
            // Ensure the cheirality constraint.
            point.bearing().dot(&a).is_sign_positive() && point.bearing().dot(&b).is_sign_positive()
        })
    }
}

/// From the paper "Closed-Form Optimal Two-View Triangulation Based on Angular Errors"
/// in section 7: "Closed-Form L∞ Triangulation".
///
/// It triangulates by minimizing the L∞ angular distance, which is the max of both angles
///
/// # Example
/// ```
/// use cv_geom::triangulation::AngularLInfinityTriangulator;
/// use cv_core::{nalgebra::{Vector3, Rotation3}, CameraToCamera, Pose, Projective, CameraPoint, TriangulatorRelative};
/// // Create a pose.
/// let pose = CameraToCamera::from_parts(Vector3::new(0.1, 0.1, 0.1), Rotation3::from_scaled_axis(Vector3::new(0.1, 0.1, 0.1)));
/// // Create a point in front of both cameras and between both cameras.
/// let real_point = CameraPoint::from_point(Vector3::new(0.3, 0.1, 2.0).into());
/// // Turn the points into bearings in each camera and try to triangulate the point again.
/// let triangulated_point = AngularLInfinityTriangulator.triangulate_relative(
///     pose,
///     real_point.bearing(),
///     pose.transform(real_point).bearing()
/// ).unwrap().point().unwrap();
/// // Verify that the point is approximately equal.
/// let real_point = real_point.point().unwrap();
/// assert!((real_point - triangulated_point).norm() < 1e-6, "real_point: {}, triangulated_point: {}", real_point, triangulated_point);
/// ```
#[derive(Copy, Clone, Debug, Default)]
pub struct AngularLInfinityTriangulator;

impl TriangulatorRelative for AngularLInfinityTriangulator {
    fn triangulate_relative(
        &self,
        original_relative_pose: CameraToCamera,
        a: UnitVector3<f64>,
        b: UnitVector3<f64>,
    ) -> Option<CameraPoint> {
        // The algorithm in the paper triangulates the point from the perspective of the second camera,
        // so this line flips and shadows everything to ensure that the triangulation happens in the first camera.
        // Technically, everything could have been reversed below, but it would be tedious and it would
        // not reflect the paper.
        let relative_pose = original_relative_pose.inverse();
        let (a, b) = (b, a);

        // Transform a into the perspective of the second camera.
        let a = relative_pose.isometry() * a;
        let translation = relative_pose.isometry().translation.vector;
        let normalized_translation = translation.normalize();
        // Correct a and b to intersect at the point which minimizes L1 distance as per
        // "Closed-Form Optimal Two-View Triangulation Based on Angular Errors" algorithm
        // 12 and 13.
        let na = (a.into_inner() + b.into_inner()).cross(&normalized_translation);
        let na_mag_squared = na.norm_squared();
        let nb = (a.into_inner() - b.into_inner()).cross(&normalized_translation);
        let nb_mag_squared = nb.norm_squared();
        let n = if na_mag_squared > nb_mag_squared {
            na / na_mag_squared.sqrt()
        } else {
            nb / nb_mag_squared.sqrt()
        };
        // Shadow the old a and b, as they have been corrected.
        let a = UnitVector3::new_normalize(a.into_inner() - (a.dot(&n) * n));
        let b = UnitVector3::new_normalize(b.into_inner() - (b.dot(&n) * n));

        let z = b.cross(&a);
        Some(CameraPoint::from_homogeneous(
            b.into_inner()
                .push(z.norm_squared() / z.dot(&translation.cross(&a))),
        ))
        .filter(|point| {
            // Ensure the point contains no NaN or infinity.
            point.homogeneous().iter().all(|n| n.is_finite())
        })
        .filter(|point| {
            // Ensure the cheirality constraint.
            point.bearing().dot(&a).is_sign_positive() && point.bearing().dot(&b).is_sign_positive()
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use cv_core::nalgebra::{IsometryMatrix3, Vector3, Vector4};

    fn old_triangulate_relative_dlt(
        tr: &RelativeDltTriangulator,
        relative_pose: CameraToCamera,
        a: UnitVector3<f64>,
        b: UnitVector3<f64>,
    ) -> Option<CameraPoint> {
        let pose = relative_pose.homogeneous();
        let mut design = Matrix4::zeros();
        design
            .row_mut(0)
            .copy_from(&RowVector4::new(-a.z, 0.0, a.x, 0.0));
        design
            .row_mut(1)
            .copy_from(&RowVector4::new(0.0, -a.z, a.y, 0.0));
        design
            .row_mut(2)
            .copy_from(&(b.x * pose.row(2) - b.z * pose.row(0)));
        design
            .row_mut(3)
            .copy_from(&(b.y * pose.row(2) - b.z * pose.row(1)));

        let svd = design.try_svd_unordered(false, true, tr.epsilon, tr.max_iterations)?;
        svd.singular_values
            .iter()
            .enumerate()
            .min_by_key(|&(_, &n)| float_ord::FloatOrd(n))
            .map(|(ix, _)| svd.v_t.unwrap().row(ix).transpose().into_owned())
            .map(CameraPoint::from_homogeneous)
            .filter(|point| point.homogeneous().iter().all(|n| n.is_finite()))
            .filter(|point| {
                point.bearing().dot(&a).is_sign_positive()
                    && point
                        .bearing()
                        .dot(&(relative_pose.isometry().inverse() * b))
                        .is_sign_positive()
            })
    }

    #[test]
    fn test_triangulate_relative_dlt() {
        for _ in 0..100 {
            let dlt = RelativeDltTriangulator::default();
            let pose = CameraToCamera(IsometryMatrix3::new(
                Vector3::new_random(),
                Vector3::new_random(),
            ));
            let real_point = CameraPoint::from_homogeneous(Vector4::new_random());
            let triangulated_point = dlt
                .triangulate_relative(
                    pose,
                    real_point.bearing(),
                    pose.transform(real_point).bearing(),
                )
                .unwrap()
                .point()
                .unwrap();
            let old_triangulated_point = old_triangulate_relative_dlt(
                &dlt,
                pose,
                real_point.bearing(),
                pose.transform(real_point).bearing(),
            )
            .unwrap()
            .point()
            .unwrap();
            assert_eq!(triangulated_point, old_triangulated_point);
        }
    }
}
