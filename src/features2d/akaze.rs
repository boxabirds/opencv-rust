use crate::core::{Mat, MatDepth};
use crate::features2d::KeyPoint;
use crate::error::{Error, Result};
use crate::core::types::{Point, Size};
use std::f64::consts::PI;

/// AKAZE (Accelerated-KAZE) detector and descriptor
/// Uses nonlinear scale space built with Fast Explicit Diffusion
pub struct AKAZE {
    pub descriptor_type: DescriptorType,
    pub descriptor_size: i32,
    pub descriptor_channels: i32,
    pub threshold: f64,
    pub n_octaves: usize,
    pub n_octave_layers: usize,
    pub diffusivity: DiffusivityType,
}

#[derive(Debug, Clone, Copy)]
pub enum DescriptorType {
    KAZE,        // Upright, not rotation invariant
    KAZEUpright, // Rotation invariant
    MLDB,        // Modified-Local Difference Binary
    MLDBUpright, // MLDB without rotation
}

#[derive(Debug, Clone, Copy)]
pub enum DiffusivityType {
    PM_G1,  // Perona-Malik, g1 = exp(-|dL|^2/k^2)
    PM_G2,  // Perona-Malik, g2 = 1/(1 + dL^2/k^2)
    WEICKERT, // Weickert diffusivity
    CHARBONNIER, // Charbonnier diffusivity
}

impl AKAZE {
    pub fn new() -> Self {
        Self {
            descriptor_type: DescriptorType::MLDB,
            descriptor_size: 0, // 0 means full size (486 bits)
            descriptor_channels: 3,
            threshold: 0.001,
            n_octaves: 4,
            n_octave_layers: 4,
            diffusivity: DiffusivityType::PM_G2,
        }
    }

    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Detect keypoints and compute descriptors
    pub fn detect_and_compute(&self, image: &Mat) -> Result<(Vec<KeyPoint>, Vec<Vec<u8>>)> {
        if image.channels() != 1 {
            return Err(Error::InvalidParameter(
                "AKAZE requires grayscale image".to_string(),
            ));
        }

        // Build nonlinear scale space
        let evolution = self.build_nonlinear_scale_space(image)?;

        // Detect keypoints using determinant of Hessian
        let keypoints = self.detect_keypoints(&evolution)?;

        // Compute M-LDB descriptors
        let descriptors = self.compute_descriptors(&evolution, &keypoints)?;

        Ok((keypoints, descriptors))
    }

    /// Build nonlinear scale space using Fast Explicit Diffusion
    fn build_nonlinear_scale_space(&self, image: &Mat) -> Result<Vec<EvolutionStep>> {
        let mut evolution = Vec::new();

        // Convert to f32 for processing
        let mut base_image = Mat::new(image.rows(), image.cols(), 1, MatDepth::F32)?;
        for row in 0..image.rows() {
            for col in 0..image.cols() {
                let pixel = base_image.at_mut(row, col)?;
                pixel[0] = (image.at(row, col)?[0] as f32) / 255.0;
            }
        }

        let sigma_0 = 1.6;
        let mut current_image = base_image.clone_mat();

        for octave in 0..self.n_octaves {
            for layer in 0..self.n_octave_layers {
                let sigma = sigma_0 * (2.0_f64).powf((layer as f64 + octave as f64 * self.n_octave_layers as f64) / self.n_octave_layers as f64);

                // Apply nonlinear diffusion
                let diffused = self.nonlinear_diffusion(&current_image, sigma)?;

                // Compute derivatives
                let lx = self.compute_derivative_x(&diffused)?;
                let ly = self.compute_derivative_y(&diffused)?;
                let lxx = self.compute_derivative_xx(&diffused)?;
                let lyy = self.compute_derivative_yy(&diffused)?;
                let lxy = self.compute_derivative_xy(&diffused)?;

                evolution.push(EvolutionStep {
                    image: diffused,
                    lx,
                    ly,
                    lxx,
                    lyy,
                    lxy,
                    sigma,
                    octave,
                    layer,
                });

                if layer == self.n_octave_layers / 2 {
                    current_image = self.half_sample(&diffused)?;
                }
            }
        }

        Ok(evolution)
    }

    /// Apply nonlinear diffusion using FED (Fast Explicit Diffusion)
    fn nonlinear_diffusion(&self, image: &Mat, sigma: f64) -> Result<Mat> {
        let mut result = image.clone_mat();

        // Number of FED cycles
        let n_cycles = ((sigma * sigma / 0.25) as usize).max(1);
        let tau = 0.25 / n_cycles as f64;

        for _ in 0..n_cycles {
            // Compute gradient
            let dx = self.compute_derivative_x(&result)?;
            let dy = self.compute_derivative_y(&result)?;

            // Compute diffusivity
            let k = 0.02; // Contrast parameter

            for row in 1..result.rows() - 1 {
                for col in 1..result.cols() - 1 {
                    let grad_mag_sq = {
                        let dx_val = dx.at(row, col)?[0];
                        let dy_val = dy.at(row, col)?[0];
                        dx_val * dx_val + dy_val * dy_val
                    };

                    let diffusivity = self.compute_diffusivity(grad_mag_sq, k);

                    // Compute divergence
                    let center = result.at(row, col)?[0];
                    let left = result.at(row, col - 1)?[0];
                    let right = result.at(row, col + 1)?[0];
                    let up = result.at(row - 1, col)?[0];
                    let down = result.at(row + 1, col)?[0];

                    let laplacian = left + right + up + down - 4.0 * center;
                    let update = diffusivity * laplacian * tau as f32;

                    let pixel = result.at_mut(row, col)?;
                    pixel[0] = (center + update).clamp(0.0, 1.0);
                }
            }
        }

        Ok(result)
    }

    fn compute_diffusivity(&self, grad_mag_sq: f32, k: f64) -> f32 {
        let k_sq = (k * k) as f32;
        match self.diffusivity {
            DiffusivityType::PM_G1 => {
                (-grad_mag_sq / k_sq).exp()
            }
            DiffusivityType::PM_G2 => {
                1.0 / (1.0 + grad_mag_sq / k_sq)
            }
            DiffusivityType::WEICKERT => {
                let lambda = 0.5;
                if grad_mag_sq == 0.0 {
                    1.0
                } else {
                    1.0 - (-(lambda / grad_mag_sq).powf(4.0)).exp()
                }
            }
            DiffusivityType::CHARBONNIER => {
                1.0 / (1.0 + grad_mag_sq / k_sq).sqrt()
            }
        }
    }

    fn compute_derivative_x(&self, image: &Mat) -> Result<Mat> {
        let mut result = Mat::new(image.rows(), image.cols(), 1, MatDepth::F32)?;

        for row in 0..image.rows() {
            for col in 1..image.cols() - 1 {
                let left = image.at(row, col - 1)?[0];
                let right = image.at(row, col + 1)?[0];
                let pixel = result.at_mut(row, col)?;
                pixel[0] = (right - left) * 0.5;
            }
        }

        Ok(result)
    }

    fn compute_derivative_y(&self, image: &Mat) -> Result<Mat> {
        let mut result = Mat::new(image.rows(), image.cols(), 1, MatDepth::F32)?;

        for row in 1..image.rows() - 1 {
            for col in 0..image.cols() {
                let up = image.at(row - 1, col)?[0];
                let down = image.at(row + 1, col)?[0];
                let pixel = result.at_mut(row, col)?;
                pixel[0] = (down - up) * 0.5;
            }
        }

        Ok(result)
    }

    fn compute_derivative_xx(&self, image: &Mat) -> Result<Mat> {
        let mut result = Mat::new(image.rows(), image.cols(), 1, MatDepth::F32)?;

        for row in 0..image.rows() {
            for col in 1..image.cols() - 1 {
                let left = image.at(row, col - 1)?[0];
                let center = image.at(row, col)?[0];
                let right = image.at(row, col + 1)?[0];
                let pixel = result.at_mut(row, col)?;
                pixel[0] = left + right - 2.0 * center;
            }
        }

        Ok(result)
    }

    fn compute_derivative_yy(&self, image: &Mat) -> Result<Mat> {
        let mut result = Mat::new(image.rows(), image.cols(), 1, MatDepth::F32)?;

        for row in 1..image.rows() - 1 {
            for col in 0..image.cols() {
                let up = image.at(row - 1, col)?[0];
                let center = image.at(row, col)?[0];
                let down = image.at(row + 1, col)?[0];
                let pixel = result.at_mut(row, col)?;
                pixel[0] = up + down - 2.0 * center;
            }
        }

        Ok(result)
    }

    fn compute_derivative_xy(&self, image: &Mat) -> Result<Mat> {
        let mut result = Mat::new(image.rows(), image.cols(), 1, MatDepth::F32)?;

        for row in 1..image.rows() - 1 {
            for col in 1..image.cols() - 1 {
                let tl = image.at(row - 1, col - 1)?[0];
                let tr = image.at(row - 1, col + 1)?[0];
                let bl = image.at(row + 1, col - 1)?[0];
                let br = image.at(row + 1, col + 1)?[0];
                let pixel = result.at_mut(row, col)?;
                pixel[0] = (br - bl - tr + tl) * 0.25;
            }
        }

        Ok(result)
    }

    fn half_sample(&self, image: &Mat) -> Result<Mat> {
        let new_rows = (image.rows() / 2).max(1);
        let new_cols = (image.cols() / 2).max(1);
        let mut result = Mat::new(new_rows, new_cols, 1, MatDepth::F32)?;

        for row in 0..new_rows {
            for col in 0..new_cols {
                let src_row = (row * 2).min(image.rows() - 1);
                let src_col = (col * 2).min(image.cols() - 1);
                let val = image.at(src_row, src_col)?[0];
                let pixel = result.at_mut(row, col)?;
                pixel[0] = val;
            }
        }

        Ok(result)
    }

    /// Detect keypoints using determinant of Hessian
    fn detect_keypoints(&self, evolution: &[EvolutionStep]) -> Result<Vec<KeyPoint>> {
        let mut keypoints = Vec::new();

        for (i, step) in evolution.iter().enumerate() {
            if i == 0 || i == evolution.len() - 1 {
                continue;
            }

            let prev = &evolution[i - 1];
            let next = &evolution[i + 1];

            for row in 5..step.image.rows() - 5 {
                for col in 5..step.image.cols() - 5 {
                    // Compute determinant of Hessian
                    let lxx = step.lxx.at(row, col)?[0];
                    let lyy = step.lyy.at(row, col)?[0];
                    let lxy = step.lxy.at(row, col)?[0];

                    let det_hessian = lxx * lyy - lxy * lxy;

                    if det_hessian > self.threshold as f32 {
                        // Check if local maximum
                        if self.is_local_maximum(det_hessian, step, prev, next, row, col)? {
                            let scale = 1 << step.octave;
                            let kp = KeyPoint {
                                pt: Point::new(col as i32 * scale, row as i32 * scale),
                                size: step.sigma as f32,
                                angle: self.compute_main_orientation(step, row, col)?,
                                response: det_hessian,
                                octave: step.octave as i32,
                            };
                            keypoints.push(kp);
                        }
                    }
                }
            }
        }

        Ok(keypoints)
    }

    fn is_local_maximum(
        &self,
        value: f32,
        curr: &EvolutionStep,
        prev: &EvolutionStep,
        next: &EvolutionStep,
        row: usize,
        col: usize,
    ) -> Result<bool> {
        // Check 3x3x3 neighborhood
        for evolution in &[prev, curr, next] {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let y = (row as i32 + dy) as usize;
                    let x = (col as i32 + dx) as usize;

                    if y >= evolution.lxx.rows() || x >= evolution.lxx.cols() {
                        continue;
                    }

                    let lxx = evolution.lxx.at(y, x)?[0];
                    let lyy = evolution.lyy.at(y, x)?[0];
                    let lxy = evolution.lxy.at(y, x)?[0];
                    let neighbor_det = lxx * lyy - lxy * lxy;

                    if neighbor_det > value {
                        return Ok(false);
                    }
                }
            }
        }

        Ok(true)
    }

    fn compute_main_orientation(&self, step: &EvolutionStep, row: usize, col: usize) -> Result<f32> {
        let mut hist = vec![0.0f32; 36]; // 10-degree bins
        let radius = 6;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let y = (row as i32 + dy).max(0).min(step.lx.rows() as i32 - 1) as usize;
                let x = (col as i32 + dx).max(0).min(step.lx.cols() as i32 - 1) as usize;

                let gx = step.lx.at(y, x)?[0];
                let gy = step.ly.at(y, x)?[0];

                let mag = (gx * gx + gy * gy).sqrt();
                let angle = gy.atan2(gx);

                let weight = mag * (-(dx * dx + dy * dy) as f32 / (2.0 * radius as f32 * radius as f32)).exp();

                let bin = (((angle * 180.0 / PI as f32 + 180.0) / 10.0) as usize) % 36;
                hist[bin] += weight;
            }
        }

        // Find dominant orientation
        let max_bin = hist
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok((max_bin as f32 * 10.0 - 180.0) * PI as f32 / 180.0)
    }

    /// Compute M-LDB (Modified Local Difference Binary) descriptors
    fn compute_descriptors(&self, evolution: &[EvolutionStep], keypoints: &[KeyPoint]) -> Result<Vec<Vec<u8>>> {
        let mut descriptors = Vec::new();

        // M-LDB sampling pattern (simplified - using grid pattern)
        let pattern_size = 10;
        let num_bits = 486; // Standard AKAZE descriptor size

        for kp in keypoints {
            let octave = kp.octave as usize;
            if octave >= evolution.len() {
                continue;
            }

            // Find appropriate scale level
            let mut step_idx = 0;
            for (i, step) in evolution.iter().enumerate() {
                if (step.sigma - kp.size as f64).abs() < 0.1 {
                    step_idx = i;
                    break;
                }
            }

            let step = &evolution[step_idx];
            let scale = 1 << step.octave;
            let row = (kp.pt.y / scale) as usize;
            let col = (kp.pt.x / scale) as usize;

            if row < pattern_size || row >= step.image.rows() - pattern_size ||
               col < pattern_size || col >= step.image.cols() - pattern_size {
                continue;
            }

            // Compute M-LDB descriptor
            let mut descriptor = vec![0u8; (num_bits + 7) / 8];
            let mut bit_idx = 0;

            let cos_angle = kp.angle.cos();
            let sin_angle = kp.angle.sin();

            // Compare pairs of points
            for i in 0..pattern_size {
                for j in 0..pattern_size {
                    if bit_idx >= num_bits {
                        break;
                    }

                    let dy1 = (i as i32 - pattern_size as i32 / 2) * 2;
                    let dx1 = (j as i32 - pattern_size as i32 / 2) * 2;

                    // Rotate coordinates
                    let ry1 = (dy1 as f32 * cos_angle - dx1 as f32 * sin_angle) as i32;
                    let rx1 = (dy1 as f32 * sin_angle + dx1 as f32 * cos_angle) as i32;

                    let y1 = (row as i32 + ry1).max(0).min(step.lx.rows() as i32 - 1) as usize;
                    let x1 = (col as i32 + rx1).max(0).min(step.lx.cols() as i32 - 1) as usize;

                    let dy2 = dy1 + 1;
                    let dx2 = dx1 + 1;

                    let ry2 = (dy2 as f32 * cos_angle - dx2 as f32 * sin_angle) as i32;
                    let rx2 = (dy2 as f32 * sin_angle + dx2 as f32 * cos_angle) as i32;

                    let y2 = (row as i32 + ry2).max(0).min(step.lx.rows() as i32 - 1) as usize;
                    let x2 = (col as i32 + rx2).max(0).min(step.lx.cols() as i32 - 1) as usize;

                    // Compare derivative responses
                    let val1 = step.lx.at(y1, x1)?[0] + step.ly.at(y1, x1)?[0];
                    let val2 = step.lx.at(y2, x2)?[0] + step.ly.at(y2, x2)?[0];

                    if val1 > val2 {
                        descriptor[bit_idx / 8] |= 1 << (bit_idx % 8);
                    }

                    bit_idx += 1;
                }
            }

            descriptors.push(descriptor);
        }

        Ok(descriptors)
    }
}

struct EvolutionStep {
    image: Mat,
    lx: Mat,
    ly: Mat,
    lxx: Mat,
    lyy: Mat,
    lxy: Mat,
    sigma: f64,
    octave: usize,
    layer: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_akaze() {
        let img = Mat::new_with_default(256, 256, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let akaze = AKAZE::new();
        let (keypoints, descriptors) = akaze.detect_and_compute(&img).unwrap();

        assert_eq!(keypoints.len(), descriptors.len());
        if !descriptors.is_empty() {
            assert_eq!(descriptors[0].len(), (486 + 7) / 8); // 61 bytes
        }
    }

    #[test]
    fn test_diffusivity() {
        let akaze = AKAZE::new();
        let diff = akaze.compute_diffusivity(0.1, 0.02);
        assert!(diff > 0.0 && diff <= 1.0);
    }
}
