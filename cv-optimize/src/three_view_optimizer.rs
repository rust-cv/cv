use argmin::{
    core::{ArgminOp, Error},
    solver::neldermead::NelderMead,
};
use average::Mean;
use core::iter::once;
use cv_core::{
    nalgebra::{Matrix6x2, UnitVector3},
    CameraToCamera, Pose, Projective, TriangulatorObservations,
};

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
