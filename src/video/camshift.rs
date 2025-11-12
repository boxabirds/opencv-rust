use crate::core::{Mat, MatDepth};
use crate::core::types::Rect;
use crate::error::{Error, Result};

/// `CAMShift` (Continuously Adaptive Mean Shift) tracker
pub struct CAMShift {
    pub term_criteria_max_iter: usize,
    pub term_criteria_epsilon: f64,
}

impl Default for CAMShift {
    fn default() -> Self {
        Self::new()
    }
}

impl CAMShift {
    #[must_use] 
    pub fn new() -> Self {
        Self {
            term_criteria_max_iter: 100,
            term_criteria_epsilon: 1.0,
        }
    }

    /// Track using `CAMShift` algorithm
    /// `prob_image`: Back-projection probability map
    /// window: Initial search window
    /// Returns: (converged window, rotation angle, number of iterations)
    pub fn track(
        &self,
        prob_image: &Mat,
        window: Rect,
    ) -> Result<(Rect, f64, usize)> {
        if prob_image.channels() != 1 {
            return Err(Error::InvalidParameter(
                "Probability image must be single-channel".to_string(),
            ));
        }

        let mut current_window = window;
        let mut iter = 0;

        // Iteratively apply mean shift
        for i in 0..self.term_criteria_max_iter {
            iter = i + 1;

            let moments = self.compute_moments(prob_image, &current_window)?;

            if moments.m00 < 1e-6 {
                break; // No content in window
            }

            // Compute center
            let cx = moments.m10 / moments.m00;
            let cy = moments.m01 / moments.m00;

            // Compute window size from second moments
            let mu20 = moments.m20 / moments.m00 - cx * cx;
            let mu02 = moments.m02 / moments.m00 - cy * cy;
            let mu11 = moments.m11 / moments.m00 - cx * cy;

            // Compute orientation
            let angle = 0.5 * (2.0 * mu11).atan2(mu20 - mu02);

            // Compute size
            let lambda1 = 0.5 * (mu20 + mu02 + ((mu20 - mu02).powi(2) + 4.0 * mu11.powi(2)).sqrt());
            let lambda2 = 0.5 * (mu20 + mu02 - ((mu20 - mu02).powi(2) + 4.0 * mu11.powi(2)).sqrt());

            #[allow(clippy::cast_possible_truncation)]
            let width = (4.0 * lambda1.sqrt()) as i32;
            #[allow(clippy::cast_possible_truncation)]
            let height = (4.0 * lambda2.sqrt()) as i32;

            // Update window
            #[allow(clippy::cast_possible_truncation)]
            let new_x = (cx - f64::from(width) / 2.0) as i32;
            #[allow(clippy::cast_possible_truncation)]
            let new_y = (cy - f64::from(height) / 2.0) as i32;

            let new_window = Rect::new(
                new_x,
                new_y,
                width.max(1),
                height.max(1),
            );

            // Check convergence
            let dx = (new_window.x - current_window.x).abs();
            let dy = (new_window.y - current_window.y).abs();
            let dw = (new_window.width - current_window.width).abs();
            let dh = (new_window.height - current_window.height).abs();

            current_window = new_window;

            #[allow(clippy::cast_possible_truncation)]
            let epsilon_i32 = self.term_criteria_epsilon as i32;
            if dx + dy + dw + dh < epsilon_i32 {
                break;
            }
        }

        // Compute final angle
        let moments = self.compute_moments(prob_image, &current_window)?;
        let angle = if moments.m00 > 1e-6 {
            let cx = moments.m10 / moments.m00;
            let cy = moments.m01 / moments.m00;
            let mu20 = moments.m20 / moments.m00 - cx * cx;
            let mu02 = moments.m02 / moments.m00 - cy * cy;
            let mu11 = moments.m11 / moments.m00 - cx * cy;
            0.5 * (2.0 * mu11).atan2(mu20 - mu02)
        } else {
            0.0
        };

        Ok((current_window, angle, iter))
    }

    fn compute_moments(&self, image: &Mat, window: &Rect) -> Result<Moments> {
        let mut moments = Moments::zero();

        #[allow(clippy::cast_sign_loss)]
        let x_start = window.x.max(0) as usize;
        #[allow(clippy::cast_sign_loss)]
        let y_start = window.y.max(0) as usize;

        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        let cols_i32 = image.cols() as i32;
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        let rows_i32 = image.rows() as i32;

        #[allow(clippy::cast_sign_loss)]
        let x_end = (window.x + window.width).min(cols_i32) as usize;
        #[allow(clippy::cast_sign_loss)]
        let y_end = (window.y + window.height).min(rows_i32) as usize;

        for y in y_start..y_end {
            for x in x_start..x_end {
                let val = f64::from(image.at(y, x)?[0]) / 255.0;

                #[allow(clippy::cast_precision_loss)]
                let x_f64 = x as f64;
                #[allow(clippy::cast_precision_loss)]
                let y_f64 = y as f64;
                #[allow(clippy::cast_precision_loss)]
                let x_start_f64 = x_start as f64;
                #[allow(clippy::cast_precision_loss)]
                let y_start_f64 = y_start as f64;

                let x_offset = x_f64 - x_start_f64;
                let y_offset = y_f64 - y_start_f64;

                moments.m00 += val;
                moments.m10 += x_offset * val;
                moments.m01 += y_offset * val;
                moments.m20 += x_offset * x_offset * val;
                moments.m11 += x_offset * y_offset * val;
                moments.m02 += y_offset * y_offset * val;
            }
        }

        // Convert back to absolute coordinates
        #[allow(clippy::cast_precision_loss)]
        let x_start_f64 = x_start as f64;
        #[allow(clippy::cast_precision_loss)]
        let y_start_f64 = y_start as f64;

        moments.m10 += x_start_f64 * moments.m00;
        moments.m01 += y_start_f64 * moments.m00;

        Ok(moments)
    }
}

struct Moments {
    m00: f64,
    m10: f64,
    m01: f64,
    m20: f64,
    m11: f64,
    m02: f64,
}

impl Moments {
    fn zero() -> Self {
        Self {
            m00: 0.0,
            m10: 0.0,
            m01: 0.0,
            m20: 0.0,
            m11: 0.0,
            m02: 0.0,
        }
    }
}

/// Dense optical flow using Farneback's algorithm
pub struct FarnebackOpticalFlow {
    pub num_levels: usize,
    pub pyr_scale: f64,
    pub fast_pyramids: bool,
    pub win_size: i32,
    pub num_iters: usize,
    pub poly_n: i32,
    pub poly_sigma: f64,
}

impl Default for FarnebackOpticalFlow {
    fn default() -> Self {
        Self::new()
    }
}

impl FarnebackOpticalFlow {
    #[must_use] 
    pub fn new() -> Self {
        Self {
            num_levels: 5,
            pyr_scale: 0.5,
            fast_pyramids: false,
            win_size: 13,
            num_iters: 10,
            poly_n: 5,
            poly_sigma: 1.1,
        }
    }

    /// Compute dense optical flow between two frames
    pub fn calc(
        &self,
        prev: &Mat,
        next: &Mat,
        flow: &mut Mat,
    ) -> Result<()> {
        if prev.channels() != 1 || next.channels() != 1 {
            return Err(Error::InvalidParameter(
                "Input images must be grayscale".to_string(),
            ));
        }

        if prev.rows() != next.rows() || prev.cols() != next.cols() {
            return Err(Error::InvalidDimensions(
                "Input images must have same dimensions".to_string(),
            ));
        }

        // Create flow field (2 channels: dx, dy)
        *flow = Mat::new(prev.rows(), prev.cols(), 2, MatDepth::F32)?;

        // Build image pyramids
        let prev_pyramid = self.build_pyramid(prev)?;
        let next_pyramid = self.build_pyramid(next)?;

        // Process from coarse to fine
        for level in (0..prev_pyramid.len()).rev() {
            self.compute_flow_level(
                &prev_pyramid[level],
                &next_pyramid[level],
                flow,
                level,
            )?;
        }

        Ok(())
    }

    fn build_pyramid(&self, image: &Mat) -> Result<Vec<Mat>> {
        let mut pyramid = vec![image.clone_mat()];

        for _ in 1..self.num_levels {
            let prev = pyramid.last().unwrap();
            #[allow(clippy::cast_precision_loss)]
            let rows_f64 = prev.rows() as f64;
            #[allow(clippy::cast_precision_loss)]
            let cols_f64 = prev.cols() as f64;

            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let new_rows = (rows_f64 * self.pyr_scale) as usize;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let new_cols = (cols_f64 * self.pyr_scale) as usize;

            if new_rows < 8 || new_cols < 8 {
                break;
            }

            let downsampled = self.downsample(prev, new_rows, new_cols)?;
            pyramid.push(downsampled);
        }

        Ok(pyramid)
    }

    fn downsample(&self, image: &Mat, new_rows: usize, new_cols: usize) -> Result<Mat> {
        let mut result = Mat::new(new_rows, new_cols, 1, image.depth())?;

        for row in 0..new_rows {
            for col in 0..new_cols {
                let src_row = (row * image.rows()) / new_rows;
                let src_col = (col * image.cols()) / new_cols;
                result.at_mut(row, col)?[0] = image.at(src_row, src_col)?[0];
            }
        }

        Ok(result)
    }

    fn compute_flow_level(
        &self,
        prev: &Mat,
        next: &Mat,
        flow: &mut Mat,
        _level: usize,
    ) -> Result<()> {
        // Simplified flow computation (full Farneback is complex)
        let half_win = self.win_size / 2;

        #[allow(clippy::cast_sign_loss)]
        let half_win_usize = half_win as usize;
        #[allow(clippy::cast_sign_loss)]
        let row_end = prev.rows() - half_win as usize;
        #[allow(clippy::cast_sign_loss)]
        let col_end = prev.cols() - half_win as usize;

        for row in half_win_usize..row_end {
            for col in half_win_usize..col_end {
                // Compute gradients
                let ix = self.compute_gradient_x(prev, row, col)?;
                let iy = self.compute_gradient_y(prev, row, col)?;
                let it = f32::from(next.at(row, col)?[0]) - f32::from(prev.at(row, col)?[0]);

                // Lucas-Kanade equation: [ix²  ix·iy] [u] = -[ix·it]
                //                          [ix·iy iy²] [v]    [iy·it]

                let ix2 = ix * ix;
                let iy2 = iy * iy;
                let ixiy = ix * iy;
                let ixit = ix * it;
                let iyit = iy * it;

                let det = ix2 * iy2 - ixiy * ixiy;

                if det.abs() > 1e-6 {
                    let u = -(iy2 * ixit - ixiy * iyit) / det;
                    let v = -(-ixiy * ixit + ix2 * iyit) / det;

                    flow.set_f32(row, col, 0, u)?;
                    flow.set_f32(row, col, 1, v)?;
                } else {
                    flow.set_f32(row, col, 0, 0.0)?;
                    flow.set_f32(row, col, 1, 0.0)?;
                }
            }
        }

        Ok(())
    }

    fn compute_gradient_x(&self, image: &Mat, row: usize, col: usize) -> Result<f32> {
        if col == 0 || col >= image.cols() - 1 {
            return Ok(0.0);
        }

        let left = f32::from(image.at(row, col - 1)?[0]);
        let right = f32::from(image.at(row, col + 1)?[0]);

        Ok((right - left) / 2.0)
    }

    fn compute_gradient_y(&self, image: &Mat, row: usize, col: usize) -> Result<f32> {
        if row == 0 || row >= image.rows() - 1 {
            return Ok(0.0);
        }

        let up = f32::from(image.at(row - 1, col)?[0]);
        let down = f32::from(image.at(row + 1, col)?[0]);

        Ok((down - up) / 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_camshift() {
        let prob = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let camshift = CAMShift::new();
        let window = Rect::new(40, 40, 20, 20);

        let (result, angle, iters) = camshift.track(&prob, window).unwrap();

        assert!(result.width > 0);
        assert!(result.height > 0);
        assert!(angle.is_finite());
        assert!(iters > 0);
    }

    #[test]
    fn test_farneback() {
        let prev = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(100.0)).unwrap();
        let next = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(110.0)).unwrap();

        let farneback = FarnebackOpticalFlow::new();
        let mut flow = Mat::new(1, 1, 1, MatDepth::F32).unwrap();

        farneback.calc(&prev, &next, &mut flow).unwrap();

        assert_eq!(flow.rows(), 50);
        assert_eq!(flow.cols(), 50);
        assert_eq!(flow.channels(), 2);
    }
}
