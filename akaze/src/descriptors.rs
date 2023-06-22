use crate::{Akaze, Error, EvolutionStep, KeyPoint};
use bitarray::BitArray;

#[cfg(feature = "rayon")]
use rayon::prelude::*;

impl Akaze {
    /// Extract descriptors from keypoints/an evolution
    ///
    /// # Arguments
    /// * `evolutions` - the nonlinear scale space
    /// * `keypoints` - the keypoints detected.
    /// * `options` - The options of the nonlinear scale space.
    /// # Return value
    /// A vector of descriptors.
    pub fn extract_descriptors(
        &self,
        evolutions: &[EvolutionStep],
        keypoints: &[KeyPoint],
    ) -> (Vec<KeyPoint>, Vec<BitArray<64>>) {
        #[cfg(not(feature = "rayon"))]
        {
            keypoints
                .iter()
                .filter_map(|&keypoint| {
                    Some((
                        keypoint,
                        self.get_mldb_descriptor(&keypoint, evolutions).ok()?,
                    ))
                })
                .unzip()
        }
        #[cfg(feature = "rayon")]
        {
            keypoints
                .par_iter()
                .filter_map(|&keypoint| {
                    Some((
                        keypoint,
                        self.get_mldb_descriptor(&keypoint, evolutions).ok()?,
                    ))
                })
                .unzip()
        }
    }

    /// Computes the rotation invariant M-LDB binary descriptor (maximum descriptor length)
    ///
    /// # Arguments
    /// `* kpt` - Input keypoint
    /// * `evolutions` - Input evolutions
    /// * `options` - Input options
    /// # Return value
    /// Binary-based descriptor
    fn get_mldb_descriptor(
        &self,
        keypoint: &KeyPoint,
        evolutions: &[EvolutionStep],
    ) -> Result<BitArray<64>, Error> {
        let mut output = BitArray::zeros();
        let max_channels = 3usize;
        debug_assert!(self.descriptor_channels <= max_channels);
        let mut values: Vec<f32> = vec![0f32; 16 * max_channels];
        let size_mult = [1.0f32, 2.0f32 / 3.0f32, 1.0f32 / 2.0f32];

        let ratio = (1u32 << keypoint.octave) as f32;
        let scale = f32::round(0.5f32 * keypoint.size / ratio);
        let xf = keypoint.point.0 / ratio;
        let yf = keypoint.point.1 / ratio;
        let co = f32::cos(keypoint.angle);
        let si = f32::sin(keypoint.angle);
        let pattern_size = self.descriptor_pattern_size as f32;

        let mut dpos = 0usize;
        for (lvl, multiplier) in size_mult.iter().enumerate() {
            let val_count = (lvl + 2usize) * (lvl + 2usize);
            let sample_size = f32::ceil(pattern_size * multiplier) as usize;
            self.mldb_fill_values(
                &mut values,
                sample_size,
                keypoint.class_id,
                xf,
                yf,
                co,
                si,
                scale,
                evolutions,
            )?;
            mldb_binary_comparisons(
                &values,
                output.bytes_mut(),
                val_count,
                &mut dpos,
                self.descriptor_channels,
            );
        }
        Ok(output)
    }

    /// Fill the comparison values for the MLDB rotation invariant descriptor
    #[allow(clippy::too_many_arguments)]
    fn mldb_fill_values(
        &self,
        values: &mut [f32],
        sample_step: usize,
        level: usize,
        xf: f32,
        yf: f32,
        co: f32,
        si: f32,
        scale: f32,
        evolutions: &[EvolutionStep],
    ) -> Result<(), Error> {
        let pattern_size = self.descriptor_pattern_size;
        let nr_channels = self.descriptor_channels;
        let mut valuepos = 0;
        for i in (-(pattern_size as i32)..(pattern_size as i32)).step_by(sample_step) {
            for j in (-(pattern_size as i32)..(pattern_size as i32)).step_by(sample_step) {
                let mut di = 0f32;
                let mut dx = 0f32;
                let mut dy = 0f32;
                let mut nsamples = 0usize;
                for k in i..(i + (sample_step as i32)) {
                    for l in j..(j + (sample_step as i32)) {
                        let l = l as f32;
                        let k = k as f32;
                        let sample_y = yf + (l * co * scale + k * si * scale);
                        let sample_x = xf + (-l * si * scale + k * co * scale);
                        let y1 = f32::round(sample_y) as isize;
                        let x1 = f32::round(sample_x) as isize;
                        if !(0..evolutions[level].Lt.width() as isize).contains(&x1)
                            || !(0..evolutions[level].Lt.height() as isize).contains(&y1)
                        {
                            return Err(Error::SampleOutOfBounds {
                                x: x1,
                                y: y1,
                                width: evolutions[level].Lt.width(),
                                height: evolutions[level].Lt.height(),
                            });
                        }
                        let y1 = y1 as usize;
                        let x1 = x1 as usize;
                        let ri = evolutions[level].Lt.get(x1, y1);
                        di += ri;
                        if nr_channels > 1 {
                            let rx = evolutions[level].Lx.get(x1, y1);
                            let ry = evolutions[level].Ly.get(x1, y1);
                            if nr_channels == 2 {
                                dx += f32::sqrt(rx * rx + ry * ry);
                            } else {
                                let rry = rx * co + ry * si;
                                let rrx = -rx * si + ry * co;
                                dx += rrx;
                                dy += rry;
                            }
                        }
                        nsamples += 1;
                    }
                }

                di /= nsamples as f32;
                dx /= nsamples as f32;
                dy /= nsamples as f32;

                values[valuepos] = di;

                if nr_channels > 1 {
                    values[valuepos + 1] = dx;
                }
                if nr_channels > 2 {
                    values[valuepos + 2] = dy;
                }
                valuepos += nr_channels;
            }
        }
        Ok(())
    }
}

/// Do the binary comparisons to obtain the descriptor
fn mldb_binary_comparisons(
    values: &[f32],
    descriptor: &mut [u8],
    count: usize,
    dpos: &mut usize,
    nr_channels: usize,
) {
    for pos in 0..nr_channels {
        for i in 0..count {
            let ival = values[nr_channels * i + pos];
            for j in (i + 1)..count {
                let res = if ival > values[nr_channels * j + pos] {
                    1u8
                } else {
                    0u8
                };
                descriptor[*dpos >> 3usize] |= res << (*dpos & 7);
                *dpos += 1usize;
            }
        }
    }
}
