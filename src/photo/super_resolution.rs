#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss, clippy::cast_precision_loss)]
use crate::core::Mat;
use crate::error::Result;

/// Super resolution using bicubic interpolation with edge enhancement
pub struct SuperResolutionBicubic {
    scale_factor: f32,
    sharpen_strength: f32,
}

impl SuperResolutionBicubic {
    #[must_use] 
    pub fn new(scale_factor: f32) -> Self {
        Self {
            scale_factor,
            sharpen_strength: 0.3,
        }
    }

    #[must_use] 
    pub fn with_sharpen(mut self, strength: f32) -> Self {
        self.sharpen_strength = strength;
        self
    }

    pub fn process(&self, src: &Mat) -> Result<Mat> {
        #[allow(clippy::cast_precision_loss)]
        let rows_f32 = src.rows() as f32;
        #[allow(clippy::cast_precision_loss)]
        let cols_f32 = src.cols() as f32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let new_rows = (rows_f32 * self.scale_factor) as usize;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let new_cols = (cols_f32 * self.scale_factor) as usize;

        // Bicubic interpolation
        let upscaled = self.bicubic_interpolation(src, new_rows, new_cols)?;

        // Edge-aware sharpening
        let sharpened = self.edge_aware_sharpen(&upscaled)?;

        Ok(sharpened)
    }

    fn bicubic_interpolation(&self, src: &Mat, new_rows: usize, new_cols: usize) -> Result<Mat> {
        let mut result = Mat::new(new_rows, new_cols, src.channels(), src.depth())?;

        #[allow(clippy::cast_precision_loss)]
        let row_scale = src.rows() as f32 / new_rows as f32;
        #[allow(clippy::cast_precision_loss)]
        let col_scale = src.cols() as f32 / new_cols as f32;

        for row in 0..new_rows {
            for col in 0..new_cols {
                #[allow(clippy::cast_precision_loss)]
                let src_row = row as f32 * row_scale;
                #[allow(clippy::cast_precision_loss)]
                let src_col = col as f32 * col_scale;

                #[allow(clippy::cast_possible_truncation)]
                for ch in 0..src.channels() {
                    let val = self.bicubic_sample(src, src_row, src_col, ch)?;
                    let clamped = val.clamp(0.0, 255.0);
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let pixel_val = clamped as u8;
                    result.at_mut(row, col)?[ch] = pixel_val;
                }
            }
        }

        Ok(result)
    }

    fn bicubic_sample(&self, src: &Mat, y: f32, x: f32, ch: usize) -> Result<f32> {
        #[allow(clippy::cast_possible_truncation)]
        let x0 = x.floor() as i32;
        #[allow(clippy::cast_possible_truncation)]
        let y0 = y.floor() as i32;
        #[allow(clippy::cast_precision_loss)]
        let x0_f32 = x0 as f32;
        #[allow(clippy::cast_precision_loss)]
        let y0_f32 = y0 as f32;
        let fx = x - x0_f32;
        let fy = y - y0_f32;

        let mut sum = 0.0f32;

        // 4x4 neighborhood for bicubic
        for j in -1..=2 {
            for i in -1..=2 {
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                let rows_max = src.rows() as i32 - 1;
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                let cols_max = src.cols() as i32 - 1;

                #[allow(clippy::cast_sign_loss)]
                let row = (y0 + j).clamp(0, rows_max) as usize;
                #[allow(clippy::cast_sign_loss)]
                let col = (x0 + i).clamp(0, cols_max) as usize;

                let pixel = f32::from(src.at(row, col)?[ch]);
                #[allow(clippy::cast_precision_loss)]
                let i_f32 = i as f32;
                #[allow(clippy::cast_precision_loss)]
                let j_f32 = j as f32;
                let weight_x = self.cubic_weight(fx - i_f32);
                let weight_y = self.cubic_weight(fy - j_f32);

                sum += pixel * weight_x * weight_y;
            }
        }

        Ok(sum)
    }

    fn cubic_weight(&self, t: f32) -> f32 {
        let a = -0.5; // Catmull-Rom spline
        let t_abs = t.abs();

        if t_abs <= 1.0 {
            (a + 2.0) * t_abs * t_abs * t_abs - (a + 3.0) * t_abs * t_abs + 1.0
        } else if t_abs < 2.0 {
            a * t_abs * t_abs * t_abs - 5.0 * a * t_abs * t_abs + 8.0 * a * t_abs - 4.0 * a
        } else {
            0.0
        }
    }

    fn edge_aware_sharpen(&self, src: &Mat) -> Result<Mat> {
        let mut result = src.clone_mat();

        // Unsharp mask
        for row in 1..src.rows() - 1 {
            for col in 1..src.cols() - 1 {
                #[allow(clippy::cast_possible_truncation)]
                for ch in 0..src.channels() {
                    let center = f32::from(src.at(row, col)?[ch]);

                    // Laplacian
                    let left = f32::from(src.at(row, col - 1)?[ch]);
                    let right = f32::from(src.at(row, col + 1)?[ch]);
                    let up = f32::from(src.at(row - 1, col)?[ch]);
                    let down = f32::from(src.at(row + 1, col)?[ch]);

                    let laplacian = left + right + up + down - 4.0 * center;

                    // Apply sharpening
                    let sharpened = center - self.sharpen_strength * laplacian;
                    let clamped = sharpened.clamp(0.0, 255.0);
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let pixel_val = clamped as u8;
                    result.at_mut(row, col)?[ch] = pixel_val;
                }
            }
        }

        Ok(result)
    }
}

/// Example-based super resolution (simplified)
pub struct SuperResolutionExample {
    scale_factor: usize,
    patch_size: usize,
}

impl SuperResolutionExample {
    #[must_use] 
    pub fn new(scale_factor: usize) -> Self {
        Self {
            scale_factor,
            patch_size: 5,
        }
    }

    pub fn process(&self, src: &Mat) -> Result<Mat> {
        let new_rows = src.rows() * self.scale_factor;
        let new_cols = src.cols() * self.scale_factor;

        let mut result = Mat::new(new_rows, new_cols, src.channels(), src.depth())?;

        // Simple patch-based upscaling
        for dst_row in 0..new_rows {
            for dst_col in 0..new_cols {
                let src_row = dst_row / self.scale_factor;
                let src_col = dst_col / self.scale_factor;

                #[allow(clippy::cast_possible_truncation)]
                for ch in 0..src.channels() {
                    let val = self.interpolate_patch(src, src_row, src_col, ch)?;
                    let clamped = val.clamp(0.0, 255.0);
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let pixel_val = clamped as u8;
                    result.at_mut(dst_row, dst_col)?[ch] = pixel_val;
                }
            }
        }

        Ok(result)
    }

    fn interpolate_patch(&self, src: &Mat, row: usize, col: usize, ch: usize) -> Result<f32> {
        let half_patch = self.patch_size / 2;
        let mut sum = 0.0f32;
        let mut count = 0;

        #[allow(clippy::cast_possible_wrap)]
        let half_patch_i32 = half_patch as i32;

        for dy in -half_patch_i32..=half_patch_i32 {
            for dx in -half_patch_i32..=half_patch_i32 {
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                let row_i32 = row as i32;
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                let col_i32 = col as i32;
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                let rows_max = src.rows() as i32 - 1;
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                let cols_max = src.cols() as i32 - 1;

                #[allow(clippy::cast_sign_loss)]
                let y = (row_i32 + dy).clamp(0, rows_max) as usize;
                #[allow(clippy::cast_sign_loss)]
                let x = (col_i32 + dx).clamp(0, cols_max) as usize;

                #[allow(clippy::cast_precision_loss)]
                let weight_val = (dx * dx + dy * dy) as f32;
                let weight = libm::expf(-weight_val / 8.0);
                sum += f32::from(src.at(y, x)?[ch]) * weight;
                count += 1;
            }
        }

        #[allow(clippy::cast_precision_loss)]
        let count_f32 = count as f32;
        Ok(sum / count_f32)
    }
}

/// Iterative back-projection super resolution
pub struct SuperResolutionBP {
    scale_factor: usize,
    iterations: usize,
    regularization: f32,
}

impl SuperResolutionBP {
    #[must_use] 
    pub fn new(scale_factor: usize) -> Self {
        Self {
            scale_factor,
            iterations: 5,
            regularization: 0.03,
        }
    }

    #[must_use] 
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn process(&self, src: &Mat) -> Result<Mat> {
        let new_rows = src.rows() * self.scale_factor;
        let new_cols = src.cols() * self.scale_factor;

        // Initial guess: bicubic upscaling
        let mut hr = self.bicubic_upsample(src, new_rows, new_cols)?;

        // Iterative back-projection
        for _ in 0..self.iterations {
            // Simulate low-resolution by downsampling
            let lr_simulated = self.downsample(&hr)?;

            // Compute error
            let error = self.compute_error(src, &lr_simulated)?;

            // Back-project error
            let error_hr = self.bicubic_upsample(&error, new_rows, new_cols)?;

            // Update high-resolution estimate
            for row in 0..hr.rows() {
                for col in 0..hr.cols() {
                    #[allow(clippy::cast_possible_truncation)]
                    for ch in 0..hr.channels() {
                        let current = f32::from(hr.at(row, col)?[ch]);
                        let correction = f32::from(error_hr.at(row, col)?[ch]);
                        let updated = current + self.regularization * correction;
                        let clamped = updated.clamp(0.0, 255.0);
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let pixel_val = clamped as u8;
                        hr.at_mut(row, col)?[ch] = pixel_val;
                    }
                }
            }
        }

        Ok(hr)
    }

    fn bicubic_upsample(&self, src: &Mat, new_rows: usize, new_cols: usize) -> Result<Mat> {
        let mut result = Mat::new(new_rows, new_cols, src.channels(), src.depth())?;

        for row in 0..new_rows {
            for col in 0..new_cols {
                let src_row = (row * src.rows()) / new_rows;
                let src_col = (col * src.cols()) / new_cols;

                #[allow(clippy::cast_possible_truncation)]
                for ch in 0..src.channels() {
                    let val = src.at(src_row.min(src.rows() - 1), src_col.min(src.cols() - 1))?[ch];
                    result.at_mut(row, col)?[ch] = val;
                }
            }
        }

        Ok(result)
    }

    fn downsample(&self, src: &Mat) -> Result<Mat> {
        let new_rows = src.rows() / self.scale_factor;
        let new_cols = src.cols() / self.scale_factor;

        let mut result = Mat::new(new_rows, new_cols, src.channels(), src.depth())?;

        for row in 0..new_rows {
            for col in 0..new_cols {
                let src_row = row * self.scale_factor;
                let src_col = col * self.scale_factor;

                #[allow(clippy::cast_possible_truncation)]
                for ch in 0..src.channels() {
                    result.at_mut(row, col)?[ch] = src.at(src_row, src_col)?[ch];
                }
            }
        }

        Ok(result)
    }

    fn compute_error(&self, original: &Mat, simulated: &Mat) -> Result<Mat> {
        let mut error = Mat::new(original.rows(), original.cols(), original.channels(), original.depth())?;

        for row in 0..original.rows() {
            for col in 0..original.cols() {
                #[allow(clippy::cast_possible_truncation)]
                for ch in 0..original.channels() {
                    let orig = i32::from(original.at(row, col)?[ch]);
                    let sim = i32::from(simulated.at(row, col)?[ch]);
                    #[allow(clippy::cast_possible_truncation)]
                    let err = (orig - sim).clamp(-255, 255) as i16;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let midpoint = i16::midpoint(err, 255) as u8;
                    error.at_mut(row, col)?[ch] = midpoint;
                }
            }
        }

        Ok(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{MatDepth, types::Scalar};

    #[test]
    fn test_super_resolution_bicubic() {
        let src = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let sr = SuperResolutionBicubic::new(2.0);
        let result = sr.process(&src).unwrap();

        assert_eq!(result.rows(), 100);
        assert_eq!(result.cols(), 100);
    }

    #[test]
    fn test_super_resolution_example() {
        let src = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let sr = SuperResolutionExample::new(2);
        let result = sr.process(&src).unwrap();

        assert_eq!(result.rows(), 100);
        assert_eq!(result.cols(), 100);
    }

    #[test]
    fn test_super_resolution_bp() {
        let src = Mat::new_with_default(25, 25, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let sr = SuperResolutionBP::new(2).with_iterations(3);
        let result = sr.process(&src).unwrap();

        assert_eq!(result.rows(), 50);
        assert_eq!(result.cols(), 50);
    }
}
