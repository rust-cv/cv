use argmin::{
    core::{ArgminOp, Error},
    solver::neldermead::NelderMead,
};
use average::Mean;
use core::iter::once;
use cv_core::{
    nalgebra::{IsometryMatrix3, Matrix6x2, Point3, Rotation3, UnitVector3, Vector3},
    CameraToCamera, Pose, Projective, TriangulatorObservations,
};
use std::ops::Add;

#[derive(Copy, Clone, Debug)]
struct Se3TangentSpace {
    translation: Vector3<f64>,
    rotation: Vector3<f64>,
}

impl Se3TangentSpace {
    fn identity() -> Self {
        Self {
            translation: Vector3::zeros(),
            rotation: Vector3::zeros(),
        }
    }

    /// Gets the isometry that represents this tangent space transformation.
    #[must_use]
    fn isometry(self) -> IsometryMatrix3<f64> {
        let rotation = Rotation3::from_scaled_axis(self.rotation);
        IsometryMatrix3::from_parts((rotation * self.translation).into(), rotation)
    }

    /// Scales both the rotation and the translation.
    #[must_use]
    fn scale(mut self, scale: f64) -> Self {
        self.translation *= scale;
        self.rotation *= scale;
        self
    }
}

impl Add for Se3TangentSpace {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            translation: self.translation + rhs.translation,
            rotation: self.rotation + rhs.rotation,
        }
    }
}

fn observation_gradient(
    point: Point3<f64>,
    bearing: UnitVector3<f64>,
    loss_cutoff: f64,
) -> Se3TangentSpace {
    let bearing = bearing.into_inner();
    let loss = 1.0 - point.coords.normalize().dot(&bearing);
    let scale = if loss < loss_cutoff {
        1.0
    } else {
        loss_cutoff / loss
    };
    // Find the distance on the observation bearing that the point projects to.
    let projection_distance = point.coords.dot(&bearing);
    // To compute the translation of the camera, we simply look at the translation needed to
    // transform the point itself into the projection of the point onto the bearing.
    // This is counter to the direction we want to move the camera, because the translation is
    // of the world in respect to the camera rather than the camera in respect to the world.
    let translation = projection_distance * bearing - point.coords;
    // Scale the point so that it would project onto the bearing at unit distance.
    // The reason we do this is so that small distances on this scale are roughly proportional to radians.
    // This is because the first order taylor approximation of `sin(x)` is `x` at `0`.
    // Since we are working with small deltas in the tangent space (SE3), this is an acceptable approximation.
    // TODO: Use loss_cutoff to create a trust region for each sample.
    let scaled = point.coords / projection_distance;
    let delta = scaled - bearing;
    // The delta's norm is now roughly in units of radians, and it points in the direction in the tangent space
    // that we wish to rotate. To compute the so(3) representation of this rotation, we need only take the cross
    // product with the bearing, and this will give us the axis on which we should rotate, with its length
    // roughly proportional to the number of radians.
    let rotation = bearing.cross(&delta);
    Se3TangentSpace {
        translation,
        rotation,
    }
    .scale(scale)
}

fn landmark_deltas(
    poses: [CameraToCamera; 2],
    observations: [UnitVector3<f64>; 3],
    triangulator: &impl TriangulatorObservations,
    loss_cutoff: f64,
) -> Option<[Se3TangentSpace; 3]> {
    let center_point = triangulator
        .triangulate_observations_to_camera(
            observations[0],
            poses.iter().copied().zip(observations[1..].iter().copied()),
        )?
        .point()?;
    let first_point = poses[0].isometry().transform_point(&center_point);
    let second_point = poses[1].isometry().transform_point(&center_point);

    Some(
        [
            (center_point, observations[0]),
            (first_point, observations[1]),
            (second_point, observations[2]),
        ]
        .map(|(point, bearing)| observation_gradient(point, bearing, loss_cutoff)),
    )
}

pub fn three_view_simple_optimize(
    mut poses: [CameraToCamera; 2],
    triangulator: &impl TriangulatorObservations,
    landmarks: &[[UnitVector3<f64>; 3]],
    loss_cutoff: f64,
    optimization_rate: f64,
    iterations: usize,
) -> [CameraToCamera; 2] {
    for _ in 0..iterations {
        let mut net_deltas = [Se3TangentSpace::identity(); 3];
        for &observations in landmarks {
            if let Some(deltas) = landmark_deltas(poses, observations, triangulator, loss_cutoff) {
                for (net, &delta) in net_deltas.iter_mut().zip(deltas.iter()) {
                    *net = *net + delta;
                }
            }
        }
        let scale = optimization_rate / landmarks.len() as f64;
        for (pose, &net_delta) in poses.iter_mut().zip(net_deltas[1..].iter()) {
            *pose = CameraToCamera(
                net_delta.scale(scale).isometry()
                    * pose.isometry()
                    * net_deltas[0].scale(scale).isometry().inverse(),
            );
        }
    }
    poses
}

pub fn three_view_nelder_mead(
    first_pose: CameraToCamera,
    second_pose: CameraToCamera,
) -> NelderMead<Matrix6x2<f64>, f64> {
    let original: Matrix6x2<f64> = Matrix6x2::from_columns(&[first_pose.se3(), second_pose.se3()]);
    let translation_scale = original
        .rows(0, 3)
        .iter()
        .map(|n| n.powi(2))
        .sum::<f64>()
        .sqrt()
        * 0.001;
    let mut variants = vec![original; 13];
    #[allow(clippy::needless_range_loop)]
    for i in 0..12 {
        if i % 6 < 3 {
            // Translation simplex must be relative to existing translation.
            variants[i].column_mut(i / 6)[i % 6] += translation_scale;
        } else {
            // Rotation simplex must be kept within a small rotation (2 pi would be a complete revolution).
            variants[i].column_mut(i / 6)[i % 6] +=
                if variants[i].column(i / 6)[i % 6] > std::f64::consts::PI {
                    -std::f64::consts::PI * 0.01
                } else {
                    std::f64::consts::PI * 0.01
                };
        }
    }
    NelderMead::new().with_initial_params(variants)
}

#[derive(Clone)]
pub struct StructurelessThreeViewOptimizer<I, T> {
    loss_cutoff: f64,
    matches: I,
    triangulator: T,
}

impl<I, T> StructurelessThreeViewOptimizer<I, T>
where
    I: Iterator<Item = [UnitVector3<f64>; 3]> + Clone,
    T: TriangulatorObservations,
{
    pub fn new(matches: I, triangulator: T) -> Self {
        Self {
            loss_cutoff: 0.001,
            matches,
            triangulator,
        }
    }

    pub fn loss_cutoff(self, loss_cutoff: f64) -> Self {
        Self {
            loss_cutoff,
            ..self
        }
    }

    pub fn residual(
        &self,
        first_pose: CameraToCamera,
        second_pose: CameraToCamera,
        c: UnitVector3<f64>,
        f: UnitVector3<f64>,
        s: UnitVector3<f64>,
    ) -> Option<[f64; 3]> {
        let cp = self.triangulator.triangulate_observations_to_camera(
            c,
            once((first_pose.isometry().into(), f)).chain(once((second_pose.isometry().into(), s))),
        )?;
        let fp = first_pose.transform(cp);
        let sp = second_pose.transform(cp);
        Some([
            1.0 - cp.bearing().dot(&c),
            1.0 - fp.bearing().dot(&f),
            1.0 - sp.bearing().dot(&s),
        ])
    }

    pub fn residuals(
        &self,
        first_pose: CameraToCamera,
        second_pose: CameraToCamera,
    ) -> impl Iterator<Item = f64> + '_ {
        self.matches.clone().flat_map(move |[c, f, s]| {
            if let Some([rc, rf, rs]) = self.residual(first_pose, second_pose, c, f, s) {
                let loss = |n: f64| {
                    if n > self.loss_cutoff {
                        self.loss_cutoff
                    } else {
                        n
                    }
                };
                once(loss(rc)).chain(once(loss(rf))).chain(once(loss(rs)))
            } else {
                once(self.loss_cutoff)
                    .chain(once(self.loss_cutoff))
                    .chain(once(self.loss_cutoff))
            }
        })
    }
}

impl<I, T> ArgminOp for StructurelessThreeViewOptimizer<I, T>
where
    I: Iterator<Item = [UnitVector3<f64>; 3]> + Clone,
    T: TriangulatorObservations + Clone,
{
    type Param = Matrix6x2<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let first_pose = Pose::from_se3(p.column(0).into());
        let second_pose = Pose::from_se3(p.column(1).into());
        let mean: Mean = self.residuals(first_pose, second_pose).collect();
        Ok(mean.mean())
    }
}
