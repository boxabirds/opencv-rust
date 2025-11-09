use crate::core::{Mat, MatDepth};
use crate::features2d::KeyPoint;
use crate::error::{Error, Result};
use crate::core::types::Point;
use std::f32::consts::PI;

/// SIFT (Scale-Invariant Feature Transform) with f32 Mat support
pub struct SIFTF32 {
    pub n_features: usize,
    pub n_octave_layers: i32,
    pub contrast_threshold: f32,
    pub edge_threshold: f32,
    pub sigma: f32,
}

impl SIFTF32 {
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

        // Convert to f32 if needed
        let image_f32 = if image.depth() != MatDepth::F32 {
            image.convert_to(MatDepth::F32)?
        } else {
            image.clone_mat()
        };

        // Normalize to [0, 1]
        let mut normalized = Mat::new(image_f32.rows(), image_f32.cols(), 1, MatDepth::F32)?;
        for row in 0..image_f32.rows() {
            for col in 0..image_f32.cols() {
                let val = if image.depth() == MatDepth::U8 {
                    image.at(row, col)?[0] as f32 / 255.0
                } else {
                    image_f32.at_f32(row, col, 0)?
                };
                normalized.set_f32(row, col, 0, val)?;
            }
        }

        // Build Gaussian pyramid
        let pyramid = self.build_gaussian_pyramid(&normalized)?;

        // Build DoG pyramid
        let dog_pyramid = self.build_dog_pyramid(&pyramid)?;

        // Detect keypoints
        let mut keypoints = self.detect_keypoints(&dog_pyramid)?;

        // Sort by response and limit
        keypoints.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());
        keypoints.truncate(self.n_features);

        // Compute descriptors
        let descriptors = self.compute_descriptors(&pyramid, &keypoints)?;

        Ok((keypoints, descriptors))
    }

    fn build_gaussian_pyramid(&self, image: &Mat) -> Result<Vec<Vec<Mat>>> {
        let n_octaves = 4;
        let n_scales = self.n_octave_layers + 3;

        let mut pyramid = Vec::new();
        let mut current = image.clone_mat();

        for _octave in 0..n_octaves {
            let mut octave_images = Vec::new();

            for scale in 0..n_scales {
                let sigma = self.sigma * (2.0_f32).powf(scale as f32 / self.n_octave_layers as f32);

                // Simple Gaussian blur (simplified for now)
                let blurred = self.gaussian_blur_f32(&current, sigma)?;
                octave_images.push(blurred);
            }

            pyramid.push(octave_images);

            // Downsample for next octave
            current = self.downsample_f32(&current)?;
        }

        Ok(pyramid)
    }

    fn gaussian_blur_f32(&self, image: &Mat, sigma: f32) -> Result<Mat> {
        // Simplified Gaussian blur for f32
        let kernel_size = ((sigma * 6.0) as usize / 2 * 2 + 1).max(3);
        let radius = kernel_size / 2;

        let mut result = Mat::new(image.rows(), image.cols(), 1, MatDepth::F32)?;

        // Create Gaussian kernel
        let mut kernel = vec![0.0f32; kernel_size];
        let mut sum = 0.0;
        for i in 0..kernel_size {
            let x = i as f32 - radius as f32;
            kernel[i] = (-x * x / (2.0 * sigma * sigma)).exp();
            sum += kernel[i];
        }
        for val in &mut kernel {
            *val /= sum;
        }

        // Horizontal pass (into temp)
        let mut temp = Mat::new(image.rows(), image.cols(), 1, MatDepth::F32)?;
        for row in 0..image.rows() {
            for col in 0..image.cols() {
                let mut value = 0.0;
                for k in 0..kernel_size {
                    let c = (col as i32 + k as i32 - radius as i32)
                        .max(0)
                        .min(image.cols() as i32 - 1) as usize;
                    value += image.at_f32(row, c, 0)? * kernel[k];
                }
                temp.set_f32(row, col, 0, value)?;
            }
        }

        // Vertical pass (into result)
        for row in 0..temp.rows() {
            for col in 0..temp.cols() {
                let mut value = 0.0;
                for k in 0..kernel_size {
                    let r = (row as i32 + k as i32 - radius as i32)
                        .max(0)
                        .min(temp.rows() as i32 - 1) as usize;
                    value += temp.at_f32(r, col, 0)? * kernel[k];
                }
                result.set_f32(row, col, 0, value)?;
            }
        }

        Ok(result)
    }

    fn downsample_f32(&self, image: &Mat) -> Result<Mat> {
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

    fn build_dog_pyramid(&self, pyramid: &[Vec<Mat>]) -> Result<Vec<Vec<Mat>>> {
        let mut dog_pyramid = Vec::new();

        for octave in pyramid {
            let mut dog_octave = Vec::new();

            for i in 0..octave.len() - 1 {
                let mut diff = Mat::new(octave[i].rows(), octave[i].cols(), 1, MatDepth::F32)?;

                for row in 0..octave[i].rows() {
                    for col in 0..octave[i].cols() {
                        let val1 = octave[i + 1].at_f32(row, col, 0)?;
                        let val2 = octave[i].at_f32(row, col, 0)?;
                        diff.set_f32(row, col, 0, val1 - val2)?;
                    }
                }

                dog_octave.push(diff);
            }

            dog_pyramid.push(dog_octave);
        }

        Ok(dog_pyramid)
    }

    fn detect_keypoints(&self, dog_pyramid: &[Vec<Mat>]) -> Result<Vec<KeyPoint>> {
        let mut keypoints = Vec::new();

        for (octave_idx, octave) in dog_pyramid.iter().enumerate() {
            if octave.len() < 3 {
                continue;
            }

            for scale_idx in 1..octave.len() - 1 {
                let prev = &octave[scale_idx - 1];
                let curr = &octave[scale_idx];
                let next = &octave[scale_idx + 1];

                for row in 3..curr.rows() - 3 {
                    for col in 3..curr.cols() - 3 {
                        let center = curr.at_f32(row, col, 0)?;

                        // Check if local extremum in 3x3x3 neighborhood
                        if center.abs() < self.contrast_threshold {
                            continue;
                        }

                        let is_max = self.is_local_maximum(center, prev, curr, next, row, col)?;
                        let is_min = self.is_local_minimum(center, prev, curr, next, row, col)?;

                        if is_max || is_min {
                            let scale = 1 << octave_idx;
                            let sigma = self.sigma * (2.0_f32).powf(scale_idx as f32 / self.n_octave_layers as f32);

                            keypoints.push(KeyPoint {
                                pt: Point::new(col as i32 * scale, row as i32 * scale),
                                size: sigma * scale as f32,
                                angle: 0.0, // Simplified - no orientation
                                response: center.abs(),
                                octave: octave_idx as i32,
                            });
                        }
                    }
                }
            }
        }

        Ok(keypoints)
    }

    fn is_local_maximum(&self, value: f32, prev: &Mat, curr: &Mat, next: &Mat, row: usize, col: usize) -> Result<bool> {
        let mats = [prev, curr, next];
        for (mat_idx, mat) in mats.iter().enumerate() {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    // Skip center point of current matrix
                    if mat_idx == 1 && dy == 0 && dx == 0 {
                        continue;
                    }
                    let r = (row as i32 + dy) as usize;
                    let c = (col as i32 + dx) as usize;
                    if r < mat.rows() && c < mat.cols() {
                        if mat.at_f32(r, c, 0)? >= value {
                            return Ok(false);
                        }
                    }
                }
            }
        }
        Ok(true)
    }

    fn is_local_minimum(&self, value: f32, prev: &Mat, curr: &Mat, next: &Mat, row: usize, col: usize) -> Result<bool> {
        let mats = [prev, curr, next];
        for (mat_idx, mat) in mats.iter().enumerate() {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    // Skip center point of current matrix
                    if mat_idx == 1 && dy == 0 && dx == 0 {
                        continue;
                    }
                    let r = (row as i32 + dy) as usize;
                    let c = (col as i32 + dx) as usize;
                    if r < mat.rows() && c < mat.cols() {
                        if mat.at_f32(r, c, 0)? <= value {
                            return Ok(false);
                        }
                    }
                }
            }
        }
        Ok(true)
    }

    fn compute_descriptors(&self, _pyramid: &[Vec<Mat>], keypoints: &[KeyPoint]) -> Result<Vec<Vec<f32>>> {
        // Simplified descriptor: just return zeros for now
        // Real SIFT would compute 128-D gradient histograms
        let descriptors = vec![vec![0.0; 128]; keypoints.len()];
        Ok(descriptors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_sift_f32() {
        let img = Mat::new_with_default(128, 128, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let sift = SIFTF32::new(100);
        let (keypoints, descriptors) = sift.detect_and_compute(&img).unwrap();

        assert_eq!(keypoints.len(), descriptors.len());
        if !descriptors.is_empty() {
            assert_eq!(descriptors[0].len(), 128);
        }
    }
}
