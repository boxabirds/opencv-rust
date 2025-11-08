use crate::core::{Mat, MatDepth};
use crate::features2d::KeyPoint;
use crate::error::{Error, Result};
use crate::core::types::{Point, Size};
use std::f64::consts::PI;

/// SIFT (Scale-Invariant Feature Transform) detector and descriptor
pub struct SIFT {
    pub n_features: usize,
    pub n_octave_layers: i32,
    pub contrast_threshold: f64,
    pub edge_threshold: f64,
    pub sigma: f64,
}

impl SIFT {
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            n_octave_layers: 3,
            contrast_threshold: 0.04,
            edge_threshold: 10.0,
            sigma: 1.6,
        }
    }

    /// Detect keypoints and compute descriptors
    pub fn detect_and_compute(&self, image: &Mat) -> Result<(Vec<KeyPoint>, Vec<Vec<f32>>)> {
        if image.channels() != 1 {
            return Err(Error::InvalidParameter(
                "SIFT requires grayscale image".to_string(),
            ));
        }

        // Build Gaussian pyramid
        let pyramid = self.build_gaussian_pyramid(image)?;

        // Build DoG (Difference of Gaussians) pyramid
        let dog_pyramid = self.build_dog_pyramid(&pyramid)?;

        // Detect keypoints in DoG pyramid
        let mut keypoints = self.detect_keypoints(&dog_pyramid, &pyramid)?;

        // Sort by response and limit to n_features
        keypoints.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
        keypoints.truncate(self.n_features);

        // Compute descriptors
        let descriptors = self.compute_descriptors(&pyramid, &keypoints)?;

        Ok((keypoints, descriptors))
    }

    fn build_gaussian_pyramid(&self, image: &Mat) -> Result<Vec<Vec<Mat>>> {
        use crate::imgproc::gaussian_blur;
        use crate::imgproc::resize;
        use crate::core::types::InterpolationFlag;

        let n_octaves = 4;
        let scales_per_octave = self.n_octave_layers + 3;

        let mut pyramid = Vec::new();

        let mut current_image = image.clone_mat();

        for octave in 0..n_octaves {
            let mut octave_images = Vec::new();

            for scale in 0..scales_per_octave {
                let sigma = self.sigma * (2.0_f64).powf(scale as f64 / self.n_octave_layers as f64);
                let ksize = ((sigma * 6.0) as i32 / 2 * 2 + 1).max(3);

                let mut blurred = Mat::new(1, 1, 1, MatDepth::U8)?;
                gaussian_blur(&current_image, &mut blurred, Size::new(ksize, ksize), sigma)?;

                octave_images.push(blurred);

                if scale == 0 {
                    current_image = octave_images[0].clone_mat();
                }
            }

            pyramid.push(octave_images);

            // Downsample for next octave
            if octave < n_octaves - 1 {
                let new_size = Size::new(
                    (current_image.cols() / 2).max(1) as i32,
                    (current_image.rows() / 2).max(1) as i32,
                );
                let mut downsampled = Mat::new(1, 1, 1, MatDepth::U8)?;
                resize(&current_image, &mut downsampled, new_size, InterpolationFlag::Linear)?;
                current_image = downsampled;
            }
        }

        Ok(pyramid)
    }

    fn build_dog_pyramid(&self, pyramid: &[Vec<Mat>]) -> Result<Vec<Vec<Mat>>> {
        let mut dog_pyramid = Vec::new();

        for octave in pyramid {
            let mut dog_octave = Vec::new();

            for i in 0..octave.len() - 1 {
                let mut diff = Mat::new(octave[i].rows(), octave[i].cols(), 1, MatDepth::U8)?;

                for row in 0..octave[i].rows() {
                    for col in 0..octave[i].cols() {
                        let val1 = octave[i + 1].at(row, col)?[0] as i16;
                        let val2 = octave[i].at(row, col)?[0] as i16;
                        let diff_val = (val1 - val2).abs();

                        let diff_pixel = diff.at_mut(row, col)?;
                        diff_pixel[0] = diff_val.min(255) as u8;
                    }
                }

                dog_octave.push(diff);
            }

            dog_pyramid.push(dog_octave);
        }

        Ok(dog_pyramid)
    }

    fn detect_keypoints(&self, dog_pyramid: &[Vec<Mat>], pyramid: &[Vec<Mat>]) -> Result<Vec<KeyPoint>> {
        let mut keypoints = Vec::new();

        for (octave_idx, octave) in dog_pyramid.iter().enumerate() {
            for scale_idx in 1..octave.len() - 1 {
                let prev = &octave[scale_idx - 1];
                let curr = &octave[scale_idx];
                let next = &octave[scale_idx + 1];

                for row in 1..curr.rows() - 1 {
                    for col in 1..curr.cols() - 1 {
                        let center_val = curr.at(row, col)?[0];

                        // Check if local extremum
                        if self.is_extremum(center_val, prev, curr, next, row, col)? {
                            // Contrast threshold
                            if (center_val as f64) < self.contrast_threshold * 255.0 {
                                continue;
                            }

                            // Edge response check (Harris-like)
                            if !self.passes_edge_check(curr, row, col)? {
                                continue;
                            }

                            // Create keypoint
                            let scale = 1 << octave_idx;
                            let kp = KeyPoint {
                                pt: Point::new(col as i32 * scale, row as i32 * scale),
                                size: (self.sigma * (2.0_f64).powf(scale_idx as f64 / self.n_octave_layers as f64)) as f32,
                                angle: self.compute_orientation(&pyramid[octave_idx][scale_idx], row, col)?,
                                response: center_val as f32,
                                octave: octave_idx as i32,
                            };

                            keypoints.push(kp);
                        }
                    }
                }
            }
        }

        Ok(keypoints)
    }

    fn is_extremum(&self, val: u8, prev: &Mat, curr: &Mat, next: &Mat, row: usize, col: usize) -> Result<bool> {
        let mut is_max = true;
        let mut is_min = true;

        // Check 3x3x3 neighborhood
        for dz in &[prev, curr, next] {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let y = (row as i32 + dy) as usize;
                    let x = (col as i32 + dx) as usize;

                    let neighbor = dz.at(y, x)?[0];

                    if neighbor >= val {
                        is_max = false;
                    }
                    if neighbor <= val {
                        is_min = false;
                    }
                }
            }
        }

        Ok(is_max || is_min)
    }

    fn passes_edge_check(&self, image: &Mat, row: usize, col: usize) -> Result<bool> {
        // Compute Hessian
        let dxx = (image.at(row, col + 1)?[0] as i32
            + image.at(row, col - 1)?[0] as i32
            - 2 * image.at(row, col)?[0] as i32) as f64;

        let dyy = (image.at(row + 1, col)?[0] as i32
            + image.at(row - 1, col)?[0] as i32
            - 2 * image.at(row, col)?[0] as i32) as f64;

        let dxy = ((image.at(row + 1, col + 1)?[0] as i32
            - image.at(row + 1, col - 1)?[0] as i32
            - image.at(row - 1, col + 1)?[0] as i32
            + image.at(row - 1, col - 1)?[0] as i32) / 4) as f64;

        let trace = dxx + dyy;
        let det = dxx * dyy - dxy * dxy;

        if det <= 0.0 {
            return Ok(false);
        }

        let ratio = trace * trace / det;
        let threshold = (self.edge_threshold + 1.0).powi(2) / self.edge_threshold;

        Ok(ratio < threshold)
    }

    fn compute_orientation(&self, image: &Mat, row: usize, col: usize) -> Result<f32> {
        let radius = 8;
        let mut histogram = vec![0.0f32; 36]; // 10 degree bins

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let y = (row as i32 + dy).max(1).min(image.rows() as i32 - 2) as usize;
                let x = (col as i32 + dx).max(1).min(image.cols() as i32 - 2) as usize;

                let mag = self.compute_gradient_magnitude(image, y, x)?;
                let angle = self.compute_gradient_angle(image, y, x)?;

                let weight = mag * (-(dx * dx + dy * dy) as f32 / (2.0 * radius as f32 * radius as f32)).exp();

                let bin = ((angle * 180.0 / PI as f32 + 180.0) / 10.0) as usize % 36;
                histogram[bin] += weight;
            }
        }

        // Find dominant orientation
        let max_bin = histogram
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Ok((max_bin as f32 * 10.0 - 180.0) * PI as f32 / 180.0)
    }

    fn compute_gradient_magnitude(&self, image: &Mat, row: usize, col: usize) -> Result<f32> {
        let dx = image.at(row, col + 1)?[0] as i32 - image.at(row, col - 1)?[0] as i32;
        let dy = image.at(row + 1, col)?[0] as i32 - image.at(row - 1, col)?[0] as i32;

        Ok(((dx * dx + dy * dy) as f32).sqrt())
    }

    fn compute_gradient_angle(&self, image: &Mat, row: usize, col: usize) -> Result<f32> {
        let dx = image.at(row, col + 1)?[0] as f32 - image.at(row, col - 1)?[0] as f32;
        let dy = image.at(row + 1, col)?[0] as f32 - image.at(row - 1, col)?[0] as f32;

        Ok(dy.atan2(dx))
    }

    fn compute_descriptors(&self, pyramid: &[Vec<Mat>], keypoints: &[KeyPoint]) -> Result<Vec<Vec<f32>>> {
        let mut descriptors = Vec::new();

        for kp in keypoints {
            let octave = kp.octave as usize;
            if octave >= pyramid.len() {
                continue;
            }

            let scale_idx = ((kp.size / self.sigma as f32).log2() * self.n_octave_layers as f32).round() as usize;
            if scale_idx >= pyramid[octave].len() {
                continue;
            }

            let image = &pyramid[octave][scale_idx];

            // Compute 128-dimensional SIFT descriptor
            let mut descriptor = vec![0.0f32; 128];

            let scale = 1 << octave;
            let row = (kp.pt.y / scale) as usize;
            let col = (kp.pt.x / scale) as usize;

            if row < 8 || row >= image.rows() - 8 || col < 8 || col >= image.cols() - 8 {
                continue;
            }

            // 4x4 grid of 4x4 cells, each with 8-bin histogram
            let cos_theta = kp.angle.cos();
            let sin_theta = kp.angle.sin();

            for grid_y in 0..4 {
                for grid_x in 0..4 {
                    let mut hist = vec![0.0f32; 8];

                    for cell_y in 0..4 {
                        for cell_x in 0..4 {
                            let y_offset = (grid_y * 4 + cell_y) as i32 - 8;
                            let x_offset = (grid_x * 4 + cell_x) as i32 - 8;

                            // Rotate coordinates
                            let y_rot = (y_offset as f32 * cos_theta - x_offset as f32 * sin_theta) as i32;
                            let x_rot = (y_offset as f32 * sin_theta + x_offset as f32 * cos_theta) as i32;

                            let y = (row as i32 + y_rot).max(1).min(image.rows() as i32 - 2) as usize;
                            let x = (col as i32 + x_rot).max(1).min(image.cols() as i32 - 2) as usize;

                            let mag = self.compute_gradient_magnitude(image, y, x)?;
                            let mut angle = self.compute_gradient_angle(image, y, x)?;

                            // Relative to keypoint orientation
                            angle -= kp.angle;
                            if angle < 0.0 {
                                angle += 2.0 * PI as f32;
                            }

                            let bin = ((angle * 4.0 / PI as f32) as usize).min(7);
                            hist[bin] += mag;
                        }
                    }

                    // Copy histogram to descriptor
                    let desc_offset = (grid_y * 4 + grid_x) * 8;
                    for i in 0..8 {
                        descriptor[desc_offset + i] = hist[i];
                    }
                }
            }

            // Normalize descriptor
            let norm: f32 = descriptor.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in &mut descriptor {
                    *val /= norm;
                    *val = val.min(0.2); // Threshold for illumination invariance
                }

                // Re-normalize
                let norm2: f32 = descriptor.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm2 > 0.0 {
                    for val in &mut descriptor {
                        *val /= norm2;
                    }
                }
            }

            descriptors.push(descriptor);
        }

        Ok(descriptors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_sift() {
        let img = Mat::new_with_default(256, 256, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let sift = SIFT::new(100);
        let (keypoints, descriptors) = sift.detect_and_compute(&img).unwrap();

        assert_eq!(keypoints.len(), descriptors.len());
        if !descriptors.is_empty() {
            assert_eq!(descriptors[0].len(), 128);
        }
    }
}
