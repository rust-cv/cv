mod single_view_optimizer;
mod three_view_optimizer;

pub use single_view_optimizer::*;
pub use three_view_optimizer::*;

use cv_core::{
    nalgebra::{IsometryMatrix3, Vector2, Vector6},
    Se3TangentSpace,
};

pub struct AdaMaxSo3Tangent {
    pub pose: IsometryMatrix3<f64>,
    pub ema_mean: Vector6<f64>,
    pub ema_variance: Vector2<f64>,
    pub last_rot_mag: f64,
    pub squared_recip_translation_trust_region: f64,
    pub squared_recip_rotation_trust_region: f64,
    pub translation_scale: f64,
    pub mean_momentum: f64,
    pub variance_momentum: f64,
    pub epsilon: f64,
}

impl AdaMaxSo3Tangent {
    pub fn new(
        pose: IsometryMatrix3<f64>,
        translation_trust_region: f64,
        rotation_trust_region: f64,
        translation_scale: f64,
    ) -> Self {
        let squared_recip_translation_trust_region =
            (translation_trust_region * translation_trust_region).recip();
        let squared_recip_rotation_trust_region =
            (rotation_trust_region * rotation_trust_region).recip();
        let ema_mean = Vector6::<f64>::zeros();
        let ema_variance = Vector2::<f64>::new(
            squared_recip_translation_trust_region,
            squared_recip_rotation_trust_region,
        );
        Self {
            pose,
            ema_mean,
            ema_variance,
            last_rot_mag: 0.0,
            squared_recip_translation_trust_region,
            squared_recip_rotation_trust_region,
            translation_scale,
            mean_momentum: 0.9,
            variance_momentum: 0.99,
            epsilon: 1e-12,
        }
    }

    // Returns true if it has reached stability.
    pub fn step(&mut self, tangent: Se3TangentSpace) -> bool {
        let squared_norms = Vector2::new(
            tangent.translation.norm_squared() * self.squared_recip_translation_trust_region,
            tangent.rotation.norm_squared() * self.squared_recip_rotation_trust_region,
        );

        // Update exponential moving average variance for this sample.
        self.ema_variance = self.ema_variance.zip_map(&squared_norms, |old, new| {
            let old_carry_over = self.variance_momentum * old;
            if new > old_carry_over {
                new
            } else {
                old_carry_over
            }
        });

        // Update exponential moving average mean for this sample.
        self.ema_mean =
            self.mean_momentum * self.ema_mean + (1.0 - self.mean_momentum) * tangent.to_vec();

        let mean = Se3TangentSpace::from_vec(self.ema_mean);
        let variance = self.ema_variance;
        // Compute the final scale.
        let scale = variance.map(|variance| (variance + self.epsilon).sqrt().recip());

        // Terminate the algorithm once it stabilizes.
        if (self.last_rot_mag - mean.rotation.norm()).abs() < self.epsilon {
            return true;
        }

        // Update the pose.
        let delta = mean
            .scale_translation(self.translation_scale * scale.x)
            .scale_rotation(scale.y);
        self.last_rot_mag = mean.rotation.norm();
        self.pose = delta.isometry() * self.pose;
        false
    }

    pub fn pose(&self) -> IsometryMatrix3<f64> {
        self.pose
    }
}
