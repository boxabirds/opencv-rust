use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};

/// Merge exposures using Debevec method to create HDR image
pub struct MergeDebevec {
    samples: usize,
}

impl MergeDebevec {
    pub fn new() -> Self {
        Self { samples: 256 }
    }

    /// Merge multiple exposures into HDR image
    /// exposures: Vec of images at different exposures
    /// times: Exposure times in seconds
    pub fn process(&self, exposures: &[Mat], times: &[f32]) -> Result<Mat> {
        if exposures.is_empty() || times.is_empty() {
            return Err(Error::InvalidParameter(
                "Need at least one exposure".to_string(),
            ));
        }

        if exposures.len() != times.len() {
            return Err(Error::InvalidParameter(
                "Number of exposures must match number of times".to_string(),
            ));
        }

        let rows = exposures[0].rows();
        let cols = exposures[0].cols();
        let channels = exposures[0].channels();

        // Create HDR image (f32 depth)
        let mut hdr = Mat::new(rows, cols, channels, MatDepth::F32)?;

        // Weight function: hat function centered at 128
        let weight = |z: u8| -> f32 {
            let z = z as f32;
            if z <= 127.0 {
                z / 127.0
            } else {
                (255.0 - z) / 127.0
            }
        };

        // Estimate camera response curve (simplified)
        let response_curve = self.estimate_response_curve(exposures, times)?;

        // Merge exposures
        for row in 0..rows {
            for col in 0..cols {
                for ch in 0..channels {
                    let mut weighted_sum = 0.0f32;
                    let mut weight_sum = 0.0f32;

                    for (i, exposure) in exposures.iter().enumerate() {
                        let pixel_val = exposure.at(row, col)?[ch];
                        let w = weight(pixel_val);

                        if w > 0.0 {
                            let radiance = response_curve[ch][pixel_val as usize] - times[i].ln();
                            weighted_sum += w * radiance;
                            weight_sum += w;
                        }
                    }

                    let hdr_val = if weight_sum > 0.0 {
                        (weighted_sum / weight_sum).exp()
                    } else {
                        0.0
                    };

                    hdr.set_f32(row, col, ch, hdr_val)?;
                }
            }
        }

        Ok(hdr)
    }

    fn estimate_response_curve(&self, exposures: &[Mat], times: &[f32]) -> Result<Vec<Vec<f32>>> {
        let channels = exposures[0].channels();
        let mut curves = vec![vec![0.0f32; 256]; channels];

        // Simplified Robertson response curve estimation
        for ch in 0..channels {
            for intensity in 0..256 {
                let mut sum = 0.0f32;
                let mut count = 0;

                // Sample across all exposures
                for (i, exposure) in exposures.iter().enumerate() {
                    for row in (0..exposure.rows()).step_by(10) {
                        for col in (0..exposure.cols()).step_by(10) {
                            let pixel_val = exposure.at(row, col)?[ch];
                            if pixel_val == intensity as u8 {
                                sum += (intensity as f32 / 255.0 * times[i]).ln();
                                count += 1;
                            }
                        }
                    }
                }

                curves[ch][intensity] = if count > 0 {
                    sum / count as f32
                } else {
                    (intensity as f32 / 255.0).ln()
                };
            }
        }

        Ok(curves)
    }
}

/// Tonemap HDR image to LDR using Reinhard method
pub struct TonemapReinhard {
    intensity: f32,
    light_adapt: f32,
    color_adapt: f32,
}

impl TonemapReinhard {
    pub fn new() -> Self {
        Self {
            intensity: 0.0,
            light_adapt: 1.0,
            color_adapt: 0.0,
        }
    }

    pub fn with_intensity(mut self, intensity: f32) -> Self {
        self.intensity = intensity;
        self
    }

    pub fn with_light_adapt(mut self, adapt: f32) -> Self {
        self.light_adapt = adapt;
        self
    }

    pub fn process(&self, hdr: &Mat) -> Result<Mat> {
        if hdr.depth() != MatDepth::F32 {
            return Err(Error::InvalidParameter(
                "Tonemap requires F32 HDR image".to_string(),
            ));
        }

        let rows = hdr.rows();
        let cols = hdr.cols();
        let channels = hdr.channels();

        let mut ldr = Mat::new(rows, cols, channels, MatDepth::U8)?;

        // Compute log-average luminance
        let mut log_sum = 0.0f32;
        let mut count = 0;

        for row in 0..rows {
            for col in 0..cols {
                let mut lum = 0.0f32;
                for ch in 0..channels {
                    let val = hdr.at_f32(row, col, ch)?;
                    lum += val;
                }
                lum /= channels as f32;

                if lum > 0.0 {
                    log_sum += (1e-6 + lum).ln();
                    count += 1;
                }
            }
        }

        let l_avg = (log_sum / count as f32).exp();
        let alpha = 0.18; // Key value

        // Apply Reinhard tone mapping
        for row in 0..rows {
            for col in 0..cols {
                for ch in 0..channels {
                    let mut val = hdr.at_f32(row, col, ch)?;

                    // Scale by key value
                    val = val * alpha / l_avg;

                    // Apply tone mapping operator
                    val = val / (1.0 + val);

                    // Gamma correction
                    val = val.powf(1.0 / 2.2);

                    // Clamp and convert to U8
                    let byte_val = (val * 255.0).clamp(0.0, 255.0) as u8;
                    ldr.at_mut(row, col)?[ch] = byte_val;
                }
            }
        }

        Ok(ldr)
    }
}

/// Tonemap HDR using Drago method
pub struct TonemapDrago {
    saturation: f32,
    bias: f32,
}

impl TonemapDrago {
    pub fn new() -> Self {
        Self {
            saturation: 1.0,
            bias: 0.85,
        }
    }

    pub fn with_bias(mut self, bias: f32) -> Self {
        self.bias = bias;
        self
    }

    pub fn process(&self, hdr: &Mat) -> Result<Mat> {
        if hdr.depth() != MatDepth::F32 {
            return Err(Error::InvalidParameter(
                "Tonemap requires F32 HDR image".to_string(),
            ));
        }

        let rows = hdr.rows();
        let cols = hdr.cols();
        let channels = hdr.channels();

        let mut ldr = Mat::new(rows, cols, channels, MatDepth::U8)?;

        // Find max luminance
        let mut max_lum = 0.0f32;
        for row in 0..rows {
            for col in 0..cols {
                let mut lum = 0.0f32;
                for ch in 0..channels {
                    lum += hdr.at_f32(row, col, ch)?;
                }
                lum /= channels as f32;
                max_lum = max_lum.max(lum);
            }
        }

        let lw_max = max_lum;
        let bias_power = (self.bias).ln() / (0.5_f32).ln();

        // Apply Drago tone mapping
        for row in 0..rows {
            for col in 0..cols {
                for ch in 0..channels {
                    let lw = hdr.at_f32(row, col, ch)?;

                    let ld = if lw > 0.0 {
                        let c1 = (lw / lw_max).ln() / (lw_max).ln();
                        let c2 = 2.0 + 8.0 * ((lw / lw_max).powf(bias_power));
                        let c3 = c1.ln() / c2.ln();
                        c3 / (10.0_f32).ln()
                    } else {
                        0.0
                    };

                    // Gamma correction
                    let val = ld.powf(1.0 / 2.2);

                    let byte_val = (val * 255.0).clamp(0.0, 255.0) as u8;
                    ldr.at_mut(row, col)?[ch] = byte_val;
                }
            }
        }

        Ok(ldr)
    }
}

/// Calibrate camera response function
pub fn calibrate_debevec(
    exposures: &[Mat],
    times: &[f32],
    samples: usize,
) -> Result<Vec<Vec<f32>>> {
    if exposures.is_empty() {
        return Err(Error::InvalidParameter("Need at least one exposure".to_string()));
    }

    let channels = exposures[0].channels();
    let mut response_curves = vec![vec![0.0f32; 256]; channels];

    // Simplified calibration using sampled pixels
    for ch in 0..channels {
        for intensity in 0..256 {
            let mut sum = 0.0f32;
            let mut count = 0;

            for (i, exposure) in exposures.iter().enumerate() {
                let step = (exposure.rows() * exposure.cols() / samples).max(1);

                for idx in (0..exposure.rows() * exposure.cols()).step_by(step) {
                    let row = idx / exposure.cols();
                    let col = idx % exposure.cols();

                    if row < exposure.rows() && col < exposure.cols() {
                        let pixel_val = exposure.at(row, col)?[ch];
                        if pixel_val == intensity as u8 {
                            sum += (intensity as f32 / 255.0 * times[i]).ln();
                            count += 1;
                        }
                    }
                }
            }

            response_curves[ch][intensity] = if count > 0 {
                sum / count as f32
            } else {
                (intensity as f32 / 255.0 + 1e-6).ln()
            };
        }
    }

    Ok(response_curves)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_merge_debevec() {
        let exp1 = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(50.0)).unwrap();
        let exp2 = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let exp3 = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(200.0)).unwrap();

        let exposures = vec![exp1, exp2, exp3];
        let times = vec![0.033, 0.066, 0.133];

        let merger = MergeDebevec::new();
        let hdr = merger.process(&exposures, &times).unwrap();

        assert_eq!(hdr.depth(), MatDepth::F32);
        assert_eq!(hdr.rows(), 50);
        assert_eq!(hdr.cols(), 50);
    }

    #[test]
    fn test_tonemap_reinhard() {
        let mut hdr = Mat::new(50, 50, 3, MatDepth::F32).unwrap();

        // Fill with sample HDR values
        for row in 0..50 {
            for col in 0..50 {
                for ch in 0..3 {
                    hdr.set_f32(row, col, ch, 2.5).unwrap();
                }
            }
        }

        let tonemap = TonemapReinhard::new();
        let ldr = tonemap.process(&hdr).unwrap();

        assert_eq!(ldr.depth(), MatDepth::U8);
        assert_eq!(ldr.rows(), 50);
    }

    #[test]
    fn test_tonemap_drago() {
        let mut hdr = Mat::new(50, 50, 3, MatDepth::F32).unwrap();

        for row in 0..50 {
            for col in 0..50 {
                for ch in 0..3 {
                    hdr.set_f32(row, col, ch, 1.5).unwrap();
                }
            }
        }

        let tonemap = TonemapDrago::new().with_bias(0.8);
        let ldr = tonemap.process(&hdr).unwrap();

        assert_eq!(ldr.depth(), MatDepth::U8);
    }

    #[test]
    fn test_calibrate_debevec() {
        let exp1 = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(50.0)).unwrap();
        let exp2 = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let exposures = vec![exp1, exp2];
        let times = vec![0.033, 0.066];

        let curves = calibrate_debevec(&exposures, &times, 100).unwrap();
        assert_eq!(curves.len(), 3); // 3 channels
        assert_eq!(curves[0].len(), 256);
    }
}
