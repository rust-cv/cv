mod single_view_optimizer;
mod three_view_optimizer;

use float_ord::FloatOrd;
use itertools::izip;
pub use single_view_optimizer::*;
pub use three_view_optimizer::*;

use cv_core::{
    nalgebra::{IsometryMatrix3, Vector2, Vector3, Vector6},
    CameraToCamera, FeatureMatch, FeatureWorldMatch, Se3TangentSpace, WorldToCamera,
};

#[derive(Clone)]
pub struct AdaMaxSo3Tangent {
    pub pose: IsometryMatrix3<f64>,
    pub ema_mean: Vector6<f64>,
    pub ema_variance: Vector2<f64>,
    pub squared_recip_translation_trust_region: f64,
    pub squared_recip_rotation_trust_region: f64,
    pub mean_momentum: f64,
    pub variance_momentum: f64,
    pub epsilon: f64,
}

impl AdaMaxSo3Tangent {
    pub fn new(
        pose: IsometryMatrix3<f64>,
        translation_trust_region: f64,
        rotation_trust_region: f64,
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
            squared_recip_translation_trust_region,
            squared_recip_rotation_trust_region,
            mean_momentum: 0.0,
            variance_momentum: 0.99,
            epsilon: 1e-12,
        }
    }

    // Returns None if it has reached stability, otherwise the delta as a tangent space.
    fn step_internal(&mut self, tangent: Se3TangentSpace) -> Option<Se3TangentSpace> {
        let squared_norms = Vector2::new(
            tangent.translation.norm_squared() * self.squared_recip_translation_trust_region,
            tangent.rotation.norm_squared() * self.squared_recip_rotation_trust_region,
        );

        // Keep the previous mean and variance to check for stabilization.
        let old_mean = self.ema_mean;
        let old_variance = self.ema_variance;

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
        if old_mean
            .zip_map(&self.ema_mean, |old, new| (old - new).abs() / new.abs())
            .iter()
            .all(|&f| f < self.epsilon)
            && old_variance
                .zip_map(&self.ema_variance, |old, new| (old - new).abs() / new.abs())
                .iter()
                .all(|&f| f < self.epsilon)
        {
            return None;
        }

        // Update the pose.
        let delta = mean.scale_translation(scale.x).scale_rotation(scale.y);
        Some(delta)
    }

    // Returns true if it has reached stability.
    //
    // In this case translation is rotation, and the translation will be rotated.
    pub fn step_rotational_translation(&mut self, tangent: Se3TangentSpace) -> bool {
        if let Some(delta) = self.step_internal(tangent) {
            let (translation_rotation, rotation) = delta.rotations();
            self.pose.rotation = rotation * self.pose.rotation;
            self.pose.translation.vector = translation_rotation * self.pose.translation.vector;
            false
        } else {
            true
        }
    }

    // Returns true if it has reached stability.
    //
    // In this case translation is linear, and can be added.
    pub fn step_linear_translation(&mut self, tangent: Se3TangentSpace) -> bool {
        if let Some(delta) = self.step_internal(tangent) {
            self.pose = delta.isometry() * self.pose;
            false
        } else {
            true
        }
    }

    pub fn pose(&self) -> IsometryMatrix3<f64> {
        self.pose
    }

    /// Returns `true` if we have stabilized.
    pub fn single_view_l1_step(&mut self, landmarks: &[FeatureWorldMatch]) -> bool {
        if landmarks.is_empty() {
            return true;
        }
        let inv_landmark_len = (landmarks.len() as f64).recip();
        let mut translations = vec![0.0; landmarks.len()];
        let mut rotations = vec![0.0; landmarks.len()];
        let mut l1 = Se3TangentSpace::identity();
        for (t, r, &landmark) in izip!(
            translations.iter_mut(),
            rotations.iter_mut(),
            landmarks.iter()
        ) {
            if let Some(tangent) =
                single_view_optimizer::landmark_delta(WorldToCamera(self.pose()), landmark)
            {
                *t = tangent.translation.norm();
                *r = tangent.rotation.norm();
                l1 += tangent.l1();
            } else {
                *t = 0.0;
                *r = 0.0;
            }
        }
        translations.sort_unstable_by_key(|&f| FloatOrd(f));
        rotations.sort_unstable_by_key(|&f| FloatOrd(f));

        let translation_scale = translations[translations.len() / 2] * inv_landmark_len;
        let rotation_scale = rotations[rotations.len() / 2] * inv_landmark_len;

        let tangent = l1
            .scale(inv_landmark_len)
            .scale_translation(translation_scale)
            .scale_rotation(rotation_scale);

        self.step_linear_translation(tangent)
    }
}
