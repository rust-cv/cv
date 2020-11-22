use argmin::{
    core::{ArgminOp, Error},
    solver::neldermead::NelderMead,
};
use average::Mean;
use cv_core::nalgebra::Vector6;
use cv_core::{Bearing, FeatureWorldMatch, Pose, Projective, WorldToCamera};
use ndarray::{s, Array1};

pub fn single_view_nelder_mead(pose: WorldToCamera) -> NelderMead<Array1<f64>, f64> {
    let original = Array1::from(pose.se3().iter().copied().collect::<Vec<f64>>());
    let translation_scale = original
        .slice(s![0..3])
        .iter()
        .map(|n| n.powi(2))
        .sum::<f64>()
        .sqrt()
        * 0.01;
    let mut variants = vec![original; 7];
    #[allow(clippy::needless_range_loop)]
    for i in 0..6 {
        if i < 3 {
            // Translation simplex must be relative to existing translation.
            variants[i][i] += translation_scale;
        } else {
            // Rotation simplex must be kept within a small rotation (2 pi would be a complete revolution).
            variants[i][i] += std::f64::consts::PI * 0.001;
        }
    }
    NelderMead::new().with_initial_params(variants)
}

#[derive(Clone)]
pub struct SingleViewConstraint<B> {
    loss_cutoff: f64,
    landmarks: Vec<FeatureWorldMatch<B>>,
}

impl<B> SingleViewConstraint<B>
where
    B: Bearing + Clone,
{
    /// Creates a ManyViewConstraint.
    ///
    /// Note that `landmarks` is an iterator over each landmark. Each landmark is an iterator over each
    /// pose presenting the occurence of that landmark in the pose's view. If the landmark doesn't appear in the
    /// view of the pose, the iterator should return `None` for that observance, and Some with the bearing otherwise.
    ///
    /// A landmark is often called a "track" by other MVG software, but the term landmark is preferred to avoid
    /// ambiguity between "camera tracking" and "a track".
    pub fn new(landmarks: Vec<FeatureWorldMatch<B>>) -> Self {
        Self {
            loss_cutoff: 0.0001,
            landmarks,
        }
    }

    pub fn loss_cutoff(self, loss_cutoff: f64) -> Self {
        Self {
            loss_cutoff,
            ..self
        }
    }

    fn loss(&self, n: f64) -> f64 {
        if n < self.loss_cutoff {
            n
        } else {
            self.loss_cutoff
        }
    }

    pub fn residuals(&self, pose: WorldToCamera) -> impl Iterator<Item = f64> + '_ {
        self.landmarks
            .iter()
            .map(move |FeatureWorldMatch(bearing, world_point)| {
                let camera_point = pose.transform(*world_point);
                self.loss(1.0 - bearing.bearing().dot(&camera_point.bearing()))
            })
    }
}

impl<B> ArgminOp for SingleViewConstraint<B>
where
    B: Bearing + Clone,
{
    type Param = Array1<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
        let pose = Pose::from_se3(Vector6::from_row_slice(
            p.as_slice().expect("param was not contiguous array"),
        ));
        let mean: Mean = self.residuals(pose).collect();
        Ok(mean.mean())
    }
}
