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
    PmG1,  // Perona-Malik, g1 = exp(-|dL|^2/k^2)
    PmG2,  // Perona-Malik, g2 = 1/(1 + dL^2/k^2)
    Weickert, // Weickert diffusivity
    Charbonnier, // Charbonnier diffusivity
}

impl KAZE {
    #[must_use] 
    pub fn new(extended: bool, upright: bool) -> Self {
        Self {
            extended,
            upright,
            threshold: 0.001,
            n_octaves: 4,
            n_octave_layers: 4,
            diffusivity: DiffusivityType::PmG2,
        }
    }

    #[must_use] 
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
                let val = f32::from(image.at(row, col)?[0]) / 255.0;
                base_image.set_f32(row, col, 0, val)?;
            }
        }

        let sigma_0 = 1.6;
        let mut current_image = base_image.clone_mat();

        for octave in 0..self.n_octaves {
            for layer in 0..self.n_octave_layers {
                // Scale space sigma calculation - precision loss acceptable in mathematical formula
                #[allow(clippy::cast_precision_loss)]
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
                    // Access the image we just pushed to evolution
                    current_image = self.half_sample(&evolution.last().unwrap().image)?;
                }
            }
        }

        Ok(evolution)
    }

    fn nonlinear_diffusion(&self, image: &Mat, sigma: f64) -> Result<Mat> {
        let mut result = image.clone_mat();

        // Number of diffusion iterations - diffusion calculation, precision loss acceptable
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n_iterations = ((sigma * sigma / 0.25) as usize).max(1);
        #[allow(clippy::cast_precision_loss)]
        let tau = 0.25 / n_iterations as f64;

        for _ in 0..n_iterations {
            let dx = self.compute_derivative_x(&result)?;
            let dy = self.compute_derivative_y(&result)?;

            let k = 0.02; // Contrast parameter

            for row in 1..result.rows() - 1 {
                for col in 1..result.cols() - 1 {
                    let grad_mag_sq = {
                        let dx_val = dx.at_f32(row, col, 0)?;
                        let dy_val = dy.at_f32(row, col, 0)?;
                        dx_val * dx_val + dy_val * dy_val
                    };

                    let diffusivity = self.compute_diffusivity(grad_mag_sq, k);

                    let center = result.at_f32(row, col, 0)?;
                    let left = result.at_f32(row, col - 1, 0)?;
                    let right = result.at_f32(row, col + 1, 0)?;
                    let up = result.at_f32(row - 1, col, 0)?;
                    let down = result.at_f32(row + 1, col, 0)?;

                    let laplacian = left + right + up + down - 4.0 * center;
                    #[allow(clippy::cast_possible_truncation)]
                    let update = diffusivity * laplacian * tau as f32;

                    result.set_f32(row, col, 0, (center + update).clamp(0.0, 1.0))?;
                }
            }
        }

        Ok(result)
    }

    fn compute_diffusivity(&self, grad_mag_sq: f32, k: f64) -> f32 {
        #[allow(clippy::cast_possible_truncation)]
        let k_sq = (k * k) as f32;
        match self.diffusivity {
            DiffusivityType::PmG1 => {
                (-grad_mag_sq / k_sq).exp()
            }
            DiffusivityType::PmG2 => {
                1.0 / (1.0 + grad_mag_sq / k_sq)
            }
            DiffusivityType::Weickert => {
                let lambda = 0.5;
                if grad_mag_sq == 0.0 {
                    1.0
                } else {
                    1.0 - (-(lambda / grad_mag_sq).powf(4.0)).exp()
                }
            }
            DiffusivityType::Charbonnier => {
                1.0 / (1.0 + grad_mag_sq / k_sq).sqrt()
            }
        }
    }

    fn compute_derivative_x(&self, image: &Mat) -> Result<Mat> {
        let mut result = Mat::new(image.rows(), image.cols(), 1, MatDepth::F32)?;

        for row in 0..image.rows() {
            for col in 1..image.cols() - 1 {
                let left = image.at_f32(row, col - 1, 0)?;
                let right = image.at_f32(row, col + 1, 0)?;
                result.set_f32(row, col, 0, (right - left) * 0.5)?;
            }
        }

        Ok(result)
    }

    fn compute_derivative_y(&self, image: &Mat) -> Result<Mat> {
        let mut result = Mat::new(image.rows(), image.cols(), 1, MatDepth::F32)?;

        for row in 1..image.rows() - 1 {
            for col in 0..image.cols() {
                let up = image.at_f32(row - 1, col, 0)?;
                let down = image.at_f32(row + 1, col, 0)?;
                result.set_f32(row, col, 0, (down - up) * 0.5)?;
            }
        }

        Ok(result)
    }

    fn compute_derivative_xx(&self, image: &Mat) -> Result<Mat> {
        let mut result = Mat::new(image.rows(), image.cols(), 1, MatDepth::F32)?;

        for row in 0..image.rows() {
            for col in 1..image.cols() - 1 {
                let left = image.at_f32(row, col - 1, 0)?;
                let center = image.at_f32(row, col, 0)?;
                let right = image.at_f32(row, col + 1, 0)?;
                result.set_f32(row, col, 0, left + right - 2.0 * center)?;
            }
        }

        Ok(result)
    }

    fn compute_derivative_yy(&self, image: &Mat) -> Result<Mat> {
        let mut result = Mat::new(image.rows(), image.cols(), 1, MatDepth::F32)?;

        for row in 1..image.rows() - 1 {
            for col in 0..image.cols() {
                let up = image.at_f32(row - 1, col, 0)?;
                let center = image.at_f32(row, col, 0)?;
                let down = image.at_f32(row + 1, col, 0)?;
                result.set_f32(row, col, 0, up + down - 2.0 * center)?;
            }
        }

        Ok(result)
    }

    fn compute_derivative_xy(&self, image: &Mat) -> Result<Mat> {
        let mut result = Mat::new(image.rows(), image.cols(), 1, MatDepth::F32)?;

        for row in 1..image.rows() - 1 {
            for col in 1..image.cols() - 1 {
                let tl = image.at_f32(row - 1, col - 1, 0)?;
                let tr = image.at_f32(row - 1, col + 1, 0)?;
                let bl = image.at_f32(row + 1, col - 1, 0)?;
                let br = image.at_f32(row + 1, col + 1, 0)?;
                result.set_f32(row, col, 0, (br - bl - tr + tl) * 0.25)?;
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
                let val = image.at_f32(src_row, src_col, 0)?;
                result.set_f32(row, col, 0, val)?;
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
                    let lxx = step.lxx.at_f32(row, col, 0)?;
                    let lyy = step.lyy.at_f32(row, col, 0)?;
                    let lxy = step.lxy.at_f32(row, col, 0)?;

                    let det_hessian = lxx * lyy - lxy * lxy;

                    #[allow(clippy::cast_possible_truncation)]
                    let threshold_f32 = self.threshold as f32;
                    if det_hessian > threshold_f32
                        && self.is_local_maximum(det_hessian, step, prev, next, row, col)? {
                            let scale = 1 << step.octave;
                            let angle = if self.upright {
                                0.0
                            } else {
                                self.compute_main_orientation(step, row, col)?
                            };

                            // Convert keypoint coordinates - scale is power of 2, so safe to convert
                            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                            let pt_x = col as i32 * scale;
                            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                            let pt_y = row as i32 * scale;
                            #[allow(clippy::cast_possible_truncation)]
                            let size = step.sigma as f32;
                            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                            let octave = step.octave as i32;

                            let kp = KeyPoint {
                                pt: Point::new(pt_x, pt_y),
                                size,
                                angle,
                                response: det_hessian,
                                octave,
                            };
                            keypoints.push(kp);
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
                    // Coordinate offset calculations for neighborhood check
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
                    let y = (row as i32 + dy) as usize;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
                    let x = (col as i32 + dx) as usize;

                    if y >= evolution.lxx.rows() || x >= evolution.lxx.cols() {
                        continue;
                    }

                    let lxx = evolution.lxx.at_f32(y, x, 0)?;
                    let lyy = evolution.lyy.at_f32(y, x, 0)?;
                    let lxy = evolution.lxy.at_f32(y, x, 0)?;
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
        let mut hist = [0.0f32; 36];
        let radius = 6;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                // Clamp coordinates to valid range for orientation histogram
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
                let y = (row as i32 + dy).max(0).min(step.lx.rows() as i32 - 1) as usize;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
                let x = (col as i32 + dx).max(0).min(step.lx.cols() as i32 - 1) as usize;

                let gx = step.lx.at_f32(y, x, 0)?;
                let gy = step.ly.at_f32(y, x, 0)?;

                let mag = (gx * gx + gy * gy).sqrt();
                let angle = gy.atan2(gx);

                // Gaussian weighting for orientation histogram
                #[allow(clippy::cast_possible_wrap, clippy::cast_precision_loss)]
                let weight = mag * (-(dx * dx + dy * dy) as f32 / (2.0 * radius as f32 * radius as f32)).exp();

                // Convert angle to histogram bin
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
                let bin = (((angle * 180.0 / PI as f32 + 180.0) / 10.0) as usize) % 36;
                hist[bin] += weight;
            }
        }

        let max_bin = hist
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map_or(0, |(idx, _)| idx);

        // Convert bin back to angle in radians
        #[allow(clippy::cast_precision_loss)]
        let angle_deg = max_bin as f32 * 10.0 - 180.0;
        #[allow(clippy::cast_possible_truncation)]
        let pi_f32 = PI as f32;
        Ok(angle_deg * pi_f32 / 180.0)
    }

    fn compute_descriptors(&self, evolution: &[EvolutionStep], keypoints: &[KeyPoint]) -> Result<Vec<Vec<f32>>> {
        let mut descriptors = Vec::new();

        let descriptor_size = if self.extended { 128 } else { 64 };

        for kp in keypoints {
            // Find appropriate scale level
            let mut step_idx = 0;
            for (i, step) in evolution.iter().enumerate() {
                if (step.sigma - f64::from(kp.size)).abs() < 0.1 {
                    step_idx = i;
                    break;
                }
            }

            let step = &evolution[step_idx];
            let scale = 1 << step.octave;
            // Convert keypoint coordinates back to current scale level
            #[allow(clippy::cast_sign_loss)]
            let row = (kp.pt.y / scale) as usize;
            #[allow(clippy::cast_sign_loss)]
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
                            // Compute offsets for descriptor grid
                            #[allow(clippy::cast_possible_wrap)]
                            let y_offset = (grid_y * subregion_size + sub_y) - pattern_size as i32;
                            #[allow(clippy::cast_possible_wrap)]
                            let x_offset = (grid_x * subregion_size + sub_x) - pattern_size as i32;

                            // Rotate for rotation invariance
                            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                            let y_rot = (y_offset as f32 * cos_angle - x_offset as f32 * sin_angle) as i32;
                            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                            let x_rot = (y_offset as f32 * sin_angle + x_offset as f32 * cos_angle) as i32;

                            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
                            let y = (row as i32 + y_rot).max(1).min(step.lx.rows() as i32 - 2) as usize;
                            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
                            let x = (col as i32 + x_rot).max(1).min(step.lx.cols() as i32 - 2) as usize;

                            let dx = step.lx.at_f32(y, x, 0)?;
                            let dy = step.ly.at_f32(y, x, 0)?;

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
