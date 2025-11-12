use crate::core::Mat;
use crate::features2d::KeyPoint;
use crate::error::{Error, Result};
use crate::core::types::Point;

/// ORB (Oriented FAST and Rotated BRIEF) feature detector and descriptor
pub struct ORB {
    pub n_features: usize,
    pub scale_factor: f32,
    pub n_levels: usize,
    pub edge_threshold: i32,
    pub first_level: i32,
    pub wta_k: i32,
    pub patch_size: i32,
    pub fast_threshold: i32,
}

impl ORB {
    #[must_use] 
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            scale_factor: 1.2,
            n_levels: 8,
            edge_threshold: 31,
            first_level: 0,
            wta_k: 2,
            patch_size: 31,
            fast_threshold: 20,
        }
    }

    #[must_use] 
    pub fn with_scale_factor(mut self, scale_factor: f32) -> Self {
        self.scale_factor = scale_factor;
        self
    }

    #[must_use] 
    pub fn with_n_levels(mut self, n_levels: usize) -> Self {
        self.n_levels = n_levels;
        self
    }

    /// Detect keypoints and compute descriptors
    pub fn detect_and_compute(&self, image: &Mat) -> Result<(Vec<KeyPoint>, Vec<Vec<u8>>)> {
        if image.channels() != 1 {
            return Err(Error::InvalidParameter(
                "ORB requires grayscale image".to_string(),
            ));
        }

        // Build image pyramid
        let pyramid = self.build_pyramid(image)?;

        // Detect FAST keypoints at each level
        let mut all_keypoints = Vec::new();

        for (level, img) in pyramid.iter().enumerate() {
            let keypoints = self.detect_fast_keypoints(img, level)?;
            all_keypoints.extend(keypoints);
        }

        // Sort by response and limit
        all_keypoints.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
        all_keypoints.truncate(self.n_features);

        // Compute orientation for each keypoint
        for kp in &mut all_keypoints {
            #[allow(clippy::cast_sign_loss)]
            let level = kp.octave as usize;
            if level < pyramid.len() {
                kp.angle = self.compute_orientation(&pyramid[level], kp)?;
            }
        }

        // Compute BRIEF descriptors
        let descriptors = self.compute_descriptors(&pyramid, &all_keypoints)?;

        Ok((all_keypoints, descriptors))
    }

    fn build_pyramid(&self, image: &Mat) -> Result<Vec<Mat>> {
        let mut pyramid = Vec::new();
        let mut current = image.clone_mat();
        pyramid.push(current.clone_mat());

        for _ in 1..self.n_levels {
            // Pyramid downsampling calculations - precision loss acceptable
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
            let new_rows = ((current.rows() as f32) / self.scale_factor) as usize;
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
            let new_cols = ((current.cols() as f32) / self.scale_factor) as usize;

            if new_rows < 3 || new_cols < 3 {
                break;
            }

            current = self.downsample(&current, new_rows, new_cols)?;
            pyramid.push(current.clone_mat());
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

    fn detect_fast_keypoints(&self, image: &Mat, level: usize) -> Result<Vec<KeyPoint>> {
        let mut keypoints = Vec::new();
        let threshold = self.fast_threshold;

        // FAST-9 detector (simplified)
        for row in 3..image.rows() - 3 {
            for col in 3..image.cols() - 3 {
                let center = i32::from(image.at(row, col)?[0]);

                // Check circle of 16 pixels
                let circle = [
                    i32::from(image.at(row - 3, col)?[0]),
                    i32::from(image.at(row - 3, col + 1)?[0]),
                    i32::from(image.at(row - 2, col + 2)?[0]),
                    i32::from(image.at(row - 1, col + 3)?[0]),
                    i32::from(image.at(row, col + 3)?[0]),
                    i32::from(image.at(row + 1, col + 3)?[0]),
                    i32::from(image.at(row + 2, col + 2)?[0]),
                    i32::from(image.at(row + 3, col + 1)?[0]),
                    i32::from(image.at(row + 3, col)?[0]),
                    i32::from(image.at(row + 3, col - 1)?[0]),
                    i32::from(image.at(row + 2, col - 2)?[0]),
                    i32::from(image.at(row + 1, col - 3)?[0]),
                    i32::from(image.at(row, col - 3)?[0]),
                    i32::from(image.at(row - 1, col - 3)?[0]),
                    i32::from(image.at(row - 2, col - 2)?[0]),
                    i32::from(image.at(row - 3, col - 1)?[0]),
                ];

                // Count brighter and darker pixels
                let mut brighter = 0;
                let mut darker = 0;

                for &pixel in &circle {
                    if pixel > center + threshold {
                        brighter += 1;
                    } else if pixel < center - threshold {
                        darker += 1;
                    }
                }

                // Need 12 contiguous pixels (simplified: just check count)
                if brighter >= 12 || darker >= 12 {
                    // Compute scale for this pyramid level
                    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                    let scale = self.scale_factor.powi(level as i32);

                    // Convert keypoint coordinates to original image scale
                    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                    let pt_x = (col as f32 * scale) as i32;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                    let pt_y = (row as f32 * scale) as i32;
                    #[allow(clippy::cast_precision_loss)]
                    let size = self.patch_size as f32 * scale;
                    #[allow(clippy::cast_precision_loss)]
                    let response = brighter.max(darker) as f32;
                    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                    let octave = level as i32;

                    keypoints.push(KeyPoint {
                        pt: Point::new(pt_x, pt_y),
                        size,
                        angle: 0.0,
                        response,
                        octave,
                    });
                }
            }
        }

        Ok(keypoints)
    }

    fn compute_orientation(&self, image: &Mat, kp: &KeyPoint) -> Result<f32> {
        let scale = self.scale_factor.powi(kp.octave);
        // Convert keypoint to current pyramid level coordinates
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
        let row = (kp.pt.y as f32 / scale) as usize;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_precision_loss)]
        let col = (kp.pt.x as f32 / scale) as usize;

        let radius = self.patch_size / 2;

        let mut m01 = 0.0f32;
        let mut m10 = 0.0f32;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                // Clamp coordinates to valid range for moment calculation
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
                let y = (row as i32 + dy).max(0).min(image.rows() as i32 - 1) as usize;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
                let x = (col as i32 + dx).max(0).min(image.cols() as i32 - 1) as usize;

                let val = f32::from(image.at(y, x)?[0]);

                #[allow(clippy::cast_precision_loss)]
                let dy_f32 = dy as f32;
                #[allow(clippy::cast_precision_loss)]
                let dx_f32 = dx as f32;
                m01 += dy_f32 * val;
                m10 += dx_f32 * val;
            }
        }

        Ok(m01.atan2(m10))
    }

    fn compute_descriptors(
        &self,
        pyramid: &[Mat],
        keypoints: &[KeyPoint],
    ) -> Result<Vec<Vec<u8>>> {
        let mut descriptors = Vec::new();

        // Predefined BRIEF test patterns
        let pattern = self.generate_test_pattern();

        for kp in keypoints {
            #[allow(clippy::cast_sign_loss)]
            let level = kp.octave as usize;
            if level >= pyramid.len() {
                continue;
            }

            let image = &pyramid[level];
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let scale = self.scale_factor.powi(level as i32);
            // Convert to pyramid level coordinates
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let row = (kp.pt.y as f32 / scale) as i32;
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let col = (kp.pt.x as f32 / scale) as i32;

            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let rows_i32 = image.rows() as i32;
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let cols_i32 = image.cols() as i32;

            if row < self.patch_size || row >= rows_i32 - self.patch_size
                || col < self.patch_size || col >= cols_i32 - self.patch_size
            {
                continue;
            }

            // Compute rotated BRIEF descriptor
            let mut descriptor = vec![0u8; 32]; // 256 bits

            let cos_angle = kp.angle.cos();
            let sin_angle = kp.angle.sin();

            for (bit_idx, &(p1, p2)) in pattern.iter().enumerate() {
                if bit_idx >= 256 {
                    break;
                }

                // Rotate test points for rotation-invariant BRIEF
                #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                let x1_rot = (p1.0 as f32 * cos_angle - p1.1 as f32 * sin_angle) as i32;
                #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                let y1_rot = (p1.0 as f32 * sin_angle + p1.1 as f32 * cos_angle) as i32;
                #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                let x2_rot = (p2.0 as f32 * cos_angle - p2.1 as f32 * sin_angle) as i32;
                #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
                let y2_rot = (p2.0 as f32 * sin_angle + p2.1 as f32 * cos_angle) as i32;

                let y1 = row + y1_rot;
                let x1 = col + x1_rot;
                let y2 = row + y2_rot;
                let x2 = col + x2_rot;

                if y1 >= 0 && y1 < rows_i32 && x1 >= 0 && x1 < cols_i32
                    && y2 >= 0 && y2 < rows_i32 && x2 >= 0 && x2 < cols_i32
                {
                    #[allow(clippy::cast_sign_loss)]
                    let val1 = image.at(y1 as usize, x1 as usize)?[0];
                    #[allow(clippy::cast_sign_loss)]
                    let val2 = image.at(y2 as usize, x2 as usize)?[0];

                    if val1 < val2 {
                        descriptor[bit_idx / 8] |= 1 << (bit_idx % 8);
                    }
                }
            }

            descriptors.push(descriptor);
        }

        Ok(descriptors)
    }

    fn generate_test_pattern(&self) -> Vec<((i32, i32), (i32, i32))> {
        let mut pattern = Vec::new();
        let half_patch = self.patch_size / 2;

        // Generate 256 test pairs (simplified pattern)
        for i in 0..256 {
            #[allow(clippy::cast_precision_loss)]
            let seed = i as f32;
            // Generate pseudo-random test pattern coordinates
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let x1 = ((seed * 13.0).sin() * half_patch as f32) as i32;
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let y1 = ((seed * 17.0).cos() * half_patch as f32) as i32;
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let x2 = ((seed * 19.0).sin() * half_patch as f32) as i32;
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let y2 = ((seed * 23.0).cos() * half_patch as f32) as i32;

            pattern.push(((x1, y1), (x2, y2)));
        }

        pattern
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{MatDepth, types::Scalar};

    #[test]
    fn test_orb_creation() {
        let orb = ORB::new(500).with_scale_factor(1.2).with_n_levels(8);

        assert_eq!(orb.n_features, 500);
        assert!((orb.scale_factor - 1.2).abs() < 1e-6);
    }

    #[test]
    fn test_orb_detect() {
        let img = Mat::new_with_default(128, 128, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let orb = ORB::new(100);
        let (keypoints, descriptors) = orb.detect_and_compute(&img).unwrap();

        assert_eq!(keypoints.len(), descriptors.len());
        if !descriptors.is_empty() {
            assert_eq!(descriptors[0].len(), 32); // 256 bits = 32 bytes
        }
    }
}
