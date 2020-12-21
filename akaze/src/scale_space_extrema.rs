use crate::evolution::EvolutionStep;
use crate::{Akaze, KeyPoint};
use log::*;
use std::f32::consts::PI;

impl Akaze {
    /// Compute scale space extrema to get the detector response.
    ///
    /// # Argument
    /// * `evolutions` - evolutions to mutate in place.
    /// * `options` - options to use.
    fn find_scale_space_extrema(&self, evolutions: &mut Vec<EvolutionStep>) -> Vec<KeyPoint> {
        let mut keypoint_cache: Vec<KeyPoint> = vec![];
        let smax = 10.0f32 * f32::sqrt(2.0f32);
        for (e_id, evolution) in evolutions.iter_mut().enumerate() {
            let w = evolution.Ldet.width();
            let h = evolution.Ldet.height();
            // maintain 5 iterators, one for the current pixel and one
            // for each cardinal pixel. Iterate through all non-border
            // pixels
            let mut x_m_iter = evolution.Ldet.iter();
            let mut x_m_i = x_m_iter.nth(w).unwrap(); // 0, 1
            let mut x_iter = evolution.Ldet.iter();
            let mut x_i = x_iter.nth(w + 1).unwrap(); // 1, 1
            let mut x_p_iter = evolution.Ldet.iter();
            let mut x_p_i = x_p_iter.nth(w + 2).unwrap(); // 2, 1
            let mut y_m_iter = evolution.Ldet.iter();
            let mut y_m_i = y_m_iter.nth(1).unwrap(); // 1, 0
            let mut y_p_iter = evolution.Ldet.iter();
            let mut y_p_i = y_p_iter.nth(2 * w + 1).unwrap(); // 1, 2
                                                              // Iterate from 1,1 to the second-to-last pixel of the second-to-last row
            for i in (w + 1)..(evolution.Ldet.len() - w - 1) {
                let x = i % w;
                let y = i / w;
                // Apply detector threshold
                if x != 0 && x != w && // do nothing for border pixels we will encounter in the iteration range
                *x_i > (self.detector_threshold as f32) &&
                *x_i > *x_p_i &&
                *x_i > *x_m_i &&
                *x_i > *y_m_i &&
                *x_i > *y_p_i
                {
                    let mut keypoint = KeyPoint {
                        response: f32::abs(*x_i),
                        size: (evolution.esigma * self.derivative_factor) as f32,
                        octave: evolution.octave as usize,
                        class_id: e_id,
                        point: (x as f32, y as f32),
                        angle: 0f32, // This is computed later - it is not needed for candidates
                    };
                    let ratio = f32::powf(2.0f32, evolution.octave as f32);
                    let sigma_size = f32::round(keypoint.size / ratio);
                    // Compare response with same and lower scale
                    let mut id_repeated = 0;
                    let mut is_repeated = false;
                    let mut is_extremum = true;
                    for (k, prev_keypoint) in keypoint_cache.iter().enumerate() {
                        if keypoint.class_id == prev_keypoint.class_id
                            || (keypoint.class_id != 0
                                && keypoint.class_id - 1 == prev_keypoint.class_id)
                        {
                            let dist = (keypoint.point.0 * ratio - prev_keypoint.point.0)
                                * (keypoint.point.0 * ratio - prev_keypoint.point.0)
                                + (keypoint.point.1 * ratio - prev_keypoint.point.1)
                                    * (keypoint.point.1 * ratio - prev_keypoint.point.1);
                            if dist <= keypoint.size * keypoint.size {
                                if keypoint.response > prev_keypoint.response {
                                    id_repeated = k;
                                    is_repeated = true;
                                } else {
                                    is_extremum = false;
                                }
                                break;
                            }
                        }
                    }
                    // Check bounds
                    if is_extremum {
                        // Check that the point is under the image limits for the descriptor computation
                        let left_x = f32::round(keypoint.point.0 - smax * sigma_size) - 1f32;
                        let right_x = f32::round(keypoint.point.0 + smax * sigma_size) + 1f32;
                        let up_y = f32::round(keypoint.point.1 - smax * sigma_size) - 1f32;
                        let down_y = f32::round(keypoint.point.1 + smax * sigma_size) + 1f32;
                        let is_out = left_x < 0f32
                            || right_x >= (w as f32)
                            || up_y < 0f32
                            || down_y >= (h as f32);
                        if !is_out {
                            keypoint.point = (
                                keypoint.point.0 * ratio + 0.5f32 * (ratio - 1.0f32),
                                keypoint.point.1 * ratio + 0.5f32 * (ratio - 1.0f32),
                            );
                            if !is_repeated {
                                keypoint_cache.push(keypoint);
                            } else {
                                keypoint_cache[id_repeated] = keypoint;
                            }
                        }
                    }
                }

                // increment iterators
                x_i = x_iter.next().unwrap();
                x_m_i = x_m_iter.next().unwrap();
                x_p_i = x_p_iter.next().unwrap();
                y_m_i = y_m_iter.next().unwrap();
                y_p_i = y_p_iter.next().unwrap();
            }
        }
        // Now filter points with the upper scale level
        let mut output_keypoints: Vec<KeyPoint> = vec![];
        for i in 0..keypoint_cache.len() {
            let mut is_repeated = false;
            let kp_i = keypoint_cache[i];
            for kp_j in &keypoint_cache[i..] {
                // Compare response with the upper scale
                if (kp_i.class_id + 1) == kp_j.class_id {
                    let dist = (kp_i.point.0 - kp_j.point.0) * (kp_i.point.0 - kp_j.point.0)
                        + (kp_i.point.1 - kp_j.point.1) * (kp_i.point.1 - kp_j.point.1);
                    if dist <= kp_i.size * kp_i.size {
                        is_repeated = true;
                        break;
                    }
                }
            }
            if !is_repeated {
                output_keypoints.push(kp_i);
            }
        }
        debug!("Extracted {} scale space extrema.", output_keypoints.len());
        output_keypoints
    }

    /// Detect keypoints in an image given a nonlinear scale space. Detects
    /// scale space extrema and performs sub-pixel refinement.
    ///
    /// # Arguments
    /// * `evolutions` - The fully-constructed non-linear scale space.
    /// * `options` - The options to use.
    /// # Return value
    /// The resulting keypoints.
    pub fn detect_keypoints(&self, evolutions: &mut Vec<EvolutionStep>) -> Vec<KeyPoint> {
        let mut keypoints = self.find_scale_space_extrema(evolutions);
        keypoints = do_subpixel_refinement(&keypoints, &evolutions);
        keypoints
    }
}

/// A 7x7 Gaussian kernel.
#[allow(clippy::excessive_precision)]
static GAUSS25: [[f32; 7usize]; 7usize] = [
    [
        0.0254_6481f32,
        0.0235_0698f32,
        0.0184_9125f32,
        0.0123_9505f32,
        0.0070_8017f32,
        0.0034_4629f32,
        0.0014_2946f32,
    ],
    [
        0.0235_0698f32,
        0.0216_9968f32,
        0.0170_6957f32,
        0.0114_4208f32,
        0.0065_3582f32,
        0.0031_8132f32,
        0.0013_1956f32,
    ],
    [
        0.0184_9125f32,
        0.0170_6957f32,
        0.0134_2740f32,
        0.0090_0066f32,
        0.0051_4126f32,
        0.0025_0252f32,
        0.0010_3800f32,
    ],
    [
        0.0123_9505f32,
        0.0114_4208f32,
        0.0090_0066f32,
        0.0060_3332f32,
        0.0034_4629f32,
        0.0016_7749f32,
        0.0006_9579f32,
    ],
    [
        0.0070_8017f32,
        0.0065_3582f32,
        0.0051_4126f32,
        0.0034_4629f32,
        0.0019_6855f32,
        0.0009_5820f32,
        0.0003_9744f32,
    ],
    [
        0.0034_4629f32,
        0.0031_8132f32,
        0.0025_0252f32,
        0.0016_7749f32,
        0.0009_5820f32,
        0.0004_6640f32,
        0.0001_9346f32,
    ],
    [
        0.0014_2946f32,
        0.0013_1956f32,
        0.0010_3800f32,
        0.0006_9579f32,
        0.0003_9744f32,
        0.0001_9346f32,
        0.0000_8024f32,
    ],
];

/// Compute the main orientation of the keypoint.
fn compute_main_orientation(keypoint: &mut KeyPoint, evolutions: &[EvolutionStep]) {
    let mut res_x: [f32; 109usize] = [0f32; 109usize];
    let mut res_y: [f32; 109usize] = [0f32; 109usize];
    let mut angs: [f32; 109usize] = [0f32; 109usize];
    let id: [usize; 13usize] = [6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6];
    let ratio = (1 << evolutions[keypoint.class_id].octave) as f32;
    let s = f32::round(0.5f32 * (keypoint.size as f32) / ratio);
    let xf = keypoint.point.0 / ratio;
    let yf = keypoint.point.1 / ratio;
    let level = keypoint.class_id;
    // Calculate derivatives responses for points within radius of 6*scale
    let mut idx = 0;
    for i in -6..=6 {
        for j in -6..=6 {
            if i * i + j * j < 36 {
                let iy = f32::round(yf + (j as f32) * s) as usize;
                let ix = f32::round(xf + (i as f32) * s) as usize;
                let gweight = GAUSS25[id[(i + 6) as usize]][id[(j + 6) as usize]];
                res_x[idx] = gweight * evolutions[level].Lx.get(ix, iy);
                res_y[idx] = gweight * evolutions[level].Ly.get(ix, iy);
                angs[idx] = res_y[idx].atan2(res_y[idx]);
                idx += 1;
            }
        }
    }
    // Loop slides pi/3 window around feature point
    let mut ang1 = 0f32;
    let mut sum_x = 0f32;
    let mut sum_y = 0f32;
    let mut max = 0f32;
    while ang1 < 2.0f32 * PI {
        let ang2 = if ang1 + PI / 3.0f32 > 2.0f32 * PI {
            ang1 - 5.0f32 * PI / 3.0f32
        } else {
            ang1 + PI / 3.0f32
        };
        ang1 += 0.15f32;
        for k in 0..109 {
            let ang = angs[k];
            if (ang1 < ang2 && ang1 < ang && ang < ang2)
                || (ang2 < ang1 && ((ang > 0f32 && ang < ang2) || (ang > ang1 && ang < 2.0 * PI)))
            {
                sum_x += res_x[k];
                sum_y += res_y[k];
            }
        }
        // if the vector produced from this window is longer than all
        // previous vectors then this forms the new dominant direction
        let val = sum_x * sum_x + sum_y * sum_y;
        if val > max {
            // store largest orientation
            max = val;
            keypoint.angle = sum_y.atan2(sum_x);
        }
    }
}

/// Do sub-pixel refinement
///
/// # Arguments
/// * `in_keypoints` - The keypoints to use.
/// * `evolutions` - The non-linear scale space.
/// # Return value
/// The resulting keypoints.
fn do_subpixel_refinement(
    in_keypoints: &[KeyPoint],
    evolutions: &[EvolutionStep],
) -> Vec<KeyPoint> {
    let mut result: Vec<KeyPoint> = vec![];
    for keypoint in in_keypoints.iter() {
        let ratio = f32::powf(2.0f32, keypoint.octave as f32);
        let x = f32::round(keypoint.point.0 / ratio) as usize;
        let y = f32::round(keypoint.point.1 / ratio) as usize;
        let x_i = evolutions[keypoint.class_id].Ldet.get(x, y);
        let x_p = evolutions[keypoint.class_id].Ldet.get(x + 1, y);
        let x_m = evolutions[keypoint.class_id].Ldet.get(x - 1, y);
        let y_p = evolutions[keypoint.class_id].Ldet.get(x, y + 1);
        let y_m = evolutions[keypoint.class_id].Ldet.get(x, y - 1);
        let x_p_y_p = evolutions[keypoint.class_id].Ldet.get(x + 1, y + 1);
        let x_p_y_m = evolutions[keypoint.class_id].Ldet.get(x + 1, y - 1);
        let x_m_y_p = evolutions[keypoint.class_id].Ldet.get(x - 1, y + 1);
        let x_m_y_m = evolutions[keypoint.class_id].Ldet.get(x - 1, y - 1);
        // Derivative
        let d_x = 0.5f32 * (x_p - x_m);
        let d_y = 0.5f32 * (y_p - y_m);
        // Hessian
        let d_xx = x_p + x_m - 2f32 * x_i;
        let d_yy = y_p + y_m - 2f32 * x_i;
        let d_xy = 0.25f32 * (x_p_y_p + x_m_y_m) - 0.25f32 * (x_p_y_m + x_m_y_p);
        #[allow(clippy::suspicious_operation_groupings)]
        let inv_det_a = (d_xx * d_yy - d_xy * d_xy).recip();
        let inv_a = [
            inv_det_a * d_yy,
            inv_det_a * -d_xy,
            inv_det_a * -d_xy,
            inv_det_a * d_xx,
        ];
        let dst = [
            -d_x * inv_a[0] + -d_y * inv_a[1],
            -d_x * inv_a[2] + -d_y * inv_a[3],
        ];
        if f32::abs(dst[0]) <= 1.0 && f32::abs(dst[1]) <= 1.0 {
            let mut keypoint_clone = *keypoint;
            keypoint_clone.point = ((x as f32) + dst[0], (y as f32) + dst[1]);
            keypoint_clone.point = (
                keypoint_clone.point.0 * ratio + 0.5f32 * (ratio - 1f32),
                keypoint_clone.point.1 * ratio + 0.5f32 * (ratio - 1f32),
            );
            result.push(keypoint_clone);
        }
    }
    debug!(
        "{}/{} remain after subpixel refinement.",
        result.len(),
        in_keypoints.len()
    );
    for mut keypoint in result.iter_mut() {
        compute_main_orientation(&mut keypoint, &evolutions);
    }
    result
}
