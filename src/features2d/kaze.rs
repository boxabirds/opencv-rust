use crate::core::{Mat, MatDepth};
use crate::features2d::KeyPoint;
use crate::error::{Error, Result};
use crate::core::types::Point;
use std::f64::consts::PI;

/// KAZE detector and descriptor using nonlinear scale space
/// KAZE uses floating-point descriptors (unlike AKAZE's binary descriptors)
pub struct KAZE {
    pub extended: bool,
    pub upright: bool,
    pub threshold: f64,
    pub n_octaves: usize,
    pub n_octave_layers: usize,
    pub diffusivity: DiffusivityType,
}

#[derive(Debug, Clone, Copy)]
pub enum DiffusivityType {
    PM_G1,  // Perona-Malik, g1 = exp(-|dL|^2/k^2)
    PM_G2,  // Perona-Malik, g2 = 1/(1 + dL^2/k^2)
    WEICKERT, // Weickert diffusivity
    CHARBONNIER, // Charbonnier diffusivity
}

impl KAZE {
    pub fn new(extended: bool, upright: bool) -> Self {
        Self {
            extended,
            upright,
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
    pub fn detect_and_compute(&self, image: &Mat) -> Result<(Vec<KeyPoint>, Vec<Vec<f32>>)> {
        if image.channels() != 1 {
            return Err(Error::InvalidParameter(
                "KAZE requires grayscale image".to_string(),
            ));
        }

        // Build nonlinear scale space
        let evolution = self.build_nonlinear_scale_space(image)?;

        // Detect keypoints
        let keypoints = self.detect_keypoints(&evolution)?;

        // Compute descriptors
        let descriptors = self.compute_descriptors(&evolution, &keypoints)?;

        Ok((keypoints, descriptors))
    }

    /// Build nonlinear scale space using nonlinear diffusion
    fn build_nonlinear_scale_space(&self, image: &Mat) -> Result<Vec<EvolutionStep>> {
        let mut evolution = Vec::new();

        // Convert to f32
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
                let sigma = sigma_0 * (2.0_f64).powf(
                    (layer as f64 + octave as f64 * self.n_octave_layers as f64) / self.n_octave_layers as f64
                );

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

    fn nonlinear_diffusion(&self, image: &Mat, sigma: f64) -> Result<Mat> {
        let mut result = image.clone_mat();

        // Number of diffusion iterations
        let n_iterations = ((sigma * sigma / 0.25) as usize).max(1);
        let tau = 0.25 / n_iterations as f64;

        for _ in 0..n_iterations {
            let dx = self.compute_derivative_x(&result)?;
            let dy = self.compute_derivative_y(&result)?;

            let k = 0.02; // Contrast parameter

            for row in 1..result.rows() - 1 {
                for col in 1..result.cols() - 1 {
                    let grad_mag_sq = {
                        let dx_val = dx.at(row, col)?[0];
                        let dy_val = dy.at(row, col)?[0];
                        dx_val * dx_val + dy_val * dy_val
                    };

                    let diffusivity = self.compute_diffusivity(grad_mag_sq, k);

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
                        if self.is_local_maximum(det_hessian, step, prev, next, row, col)? {
                            let scale = 1 << step.octave;
                            let angle = if self.upright {
                                0.0
                            } else {
                                self.compute_main_orientation(step, row, col)?
                            };

                            let kp = KeyPoint {
                                pt: Point::new(col as i32 * scale, row as i32 * scale),
                                size: step.sigma as f32,
                                angle,
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
        let mut hist = vec![0.0f32; 36];
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

        let max_bin = hist
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok((max_bin as f32 * 10.0 - 180.0) * PI as f32 / 180.0)
    }

    fn compute_descriptors(&self, evolution: &[EvolutionStep], keypoints: &[KeyPoint]) -> Result<Vec<Vec<f32>>> {
        let mut descriptors = Vec::new();

        let descriptor_size = if self.extended { 128 } else { 64 };

        for kp in keypoints {
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

            let pattern_size = 10;

            if row < pattern_size || row >= step.image.rows() - pattern_size ||
               col < pattern_size || col >= step.image.cols() - pattern_size {
                continue;
            }

            // Compute SURF-like descriptor
            let mut descriptor = vec![0.0f32; descriptor_size];

            let cos_angle = kp.angle.cos();
            let sin_angle = kp.angle.sin();

            let grid_size = if self.extended { 4 } else { 4 };
            let subregion_size = 5;

            let mut desc_idx = 0;

            for grid_y in 0..grid_size {
                for grid_x in 0..grid_size {
                    let mut dx_sum = 0.0f32;
                    let mut dy_sum = 0.0f32;
                    let mut abs_dx_sum = 0.0f32;
                    let mut abs_dy_sum = 0.0f32;

                    for sub_y in 0..subregion_size {
                        for sub_x in 0..subregion_size {
                            let y_offset = (grid_y * subregion_size + sub_y) as i32 - pattern_size as i32;
                            let x_offset = (grid_x * subregion_size + sub_x) as i32 - pattern_size as i32;

                            // Rotate
                            let y_rot = (y_offset as f32 * cos_angle - x_offset as f32 * sin_angle) as i32;
                            let x_rot = (y_offset as f32 * sin_angle + x_offset as f32 * cos_angle) as i32;

                            let y = (row as i32 + y_rot).max(1).min(step.lx.rows() as i32 - 2) as usize;
                            let x = (col as i32 + x_rot).max(1).min(step.lx.cols() as i32 - 2) as usize;

                            let dx = step.lx.at(y, x)?[0];
                            let dy = step.ly.at(y, x)?[0];

                            // Rotate gradient
                            let dx_rot = dx * cos_angle - dy * sin_angle;
                            let dy_rot = dx * sin_angle + dy * cos_angle;

                            dx_sum += dx_rot;
                            dy_sum += dy_rot;
                            abs_dx_sum += dx_rot.abs();
                            abs_dy_sum += dy_rot.abs();
                        }
                    }

                    if desc_idx < descriptor_size {
                        descriptor[desc_idx] = dx_sum;
                        desc_idx += 1;
                    }
                    if desc_idx < descriptor_size {
                        descriptor[desc_idx] = dy_sum;
                        desc_idx += 1;
                    }
                    if desc_idx < descriptor_size {
                        descriptor[desc_idx] = abs_dx_sum;
                        desc_idx += 1;
                    }
                    if desc_idx < descriptor_size {
                        descriptor[desc_idx] = abs_dy_sum;
                        desc_idx += 1;
                    }
                }
            }

            // Normalize
            let norm: f32 = descriptor.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut descriptor {
                    *val /= norm;
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
    fn test_kaze() {
        let img = Mat::new_with_default(256, 256, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let kaze = KAZE::new(false, false);
        let (keypoints, descriptors) = kaze.detect_and_compute(&img).unwrap();

        assert_eq!(keypoints.len(), descriptors.len());
        if !descriptors.is_empty() {
            assert_eq!(descriptors[0].len(), 64); // Standard descriptor size
        }
    }

    #[test]
    fn test_kaze_extended() {
        let img = Mat::new_with_default(256, 256, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let kaze = KAZE::new(true, false);
        let (keypoints, descriptors) = kaze.detect_and_compute(&img).unwrap();

        if !descriptors.is_empty() {
            assert_eq!(descriptors[0].len(), 128); // Extended descriptor
        }
    }

    #[test]
    fn test_diffusivity() {
        let kaze = KAZE::new(false, false);
        let diff = kaze.compute_diffusivity(0.1, 0.02);
        assert!(diff > 0.0 && diff <= 1.0);
    }
}
