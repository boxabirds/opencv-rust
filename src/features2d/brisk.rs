use crate::core::{Mat, MatDepth};
use crate::features2d::KeyPoint;
use crate::error::{Error, Result};
use crate::core::types::Point;
use std::f64::consts::PI;

/// BRISK (Binary Robust Invariant Scalable Keypoints) detector and descriptor
pub struct BRISK {
    pub threshold: i32,
    pub octaves: usize,
    pub pattern_scale: f32,
}

impl BRISK {
    #[must_use] 
    pub fn new(threshold: i32, octaves: usize) -> Self {
        Self {
            threshold,
            octaves,
            pattern_scale: 1.0,
        }
    }

    #[must_use] 
    pub fn with_pattern_scale(mut self, scale: f32) -> Self {
        self.pattern_scale = scale;
        self
    }

    /// Detect keypoints and compute descriptors
    pub fn detect_and_compute(&self, image: &Mat) -> Result<(Vec<KeyPoint>, Vec<Vec<u8>>)> {
        if image.channels() != 1 {
            return Err(Error::InvalidParameter(
                "BRISK requires grayscale image".to_string(),
            ));
        }

        // Build scale-space pyramid
        let pyramid = self.build_scale_space(image)?;

        // Detect keypoints using FAST-like detection in scale space
        let keypoints = self.detect_keypoints(&pyramid)?;

        // Compute BRISK descriptors
        let descriptors = self.compute_descriptors(image, &keypoints)?;

        Ok((keypoints, descriptors))
    }

    /// Build scale-space pyramid
    fn build_scale_space(&self, image: &Mat) -> Result<Vec<ScaleLevel>> {
        use crate::imgproc::gaussian_blur;
        use crate::core::types::{Size, InterpolationFlag};
        use crate::imgproc::resize;

        let mut pyramid = Vec::new();

        let mut current_image = image.clone_mat();

        for octave in 0..self.octaves {
            let scale = 1 << octave;

            // Intra-octave levels
            for intra in 0..4 {
                let sigma = 1.0 * (2.0_f64).powf(intra as f64 / 4.0);

                let mut blurred = Mat::new(1, 1, 1, MatDepth::U8)?;
                let ksize = ((sigma * 3.0) as i32 * 2 + 1).max(3);
                gaussian_blur(&current_image, &mut blurred, Size::new(ksize, ksize), sigma)?;

                pyramid.push(ScaleLevel {
                    image: blurred,
                    scale: scale as f32 * (2.0_f32).powf(intra as f32 / 4.0),
                    octave,
                    layer: intra,
                });
            }

            // Downsample for next octave
            if octave < self.octaves - 1 {
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

    /// Detect keypoints using AGAST-like corner detection
    fn detect_keypoints(&self, pyramid: &[ScaleLevel]) -> Result<Vec<KeyPoint>> {
        let mut keypoints = Vec::new();

        for level in pyramid {
            let image = &level.image;

            for row in 3..image.rows() - 3 {
                for col in 3..image.cols() - 3 {
                    let center = image.at(row, col)?[0];

                    // AGAST pattern (simplified)
                    let mut brighter = 0;
                    let mut darker = 0;

                    let circle = [
                        (row - 3, col),
                        (row - 3, col + 1),
                        (row - 2, col + 2),
                        (row - 1, col + 3),
                        (row, col + 3),
                        (row + 1, col + 3),
                        (row + 2, col + 2),
                        (row + 3, col + 1),
                        (row + 3, col),
                        (row + 3, col - 1),
                        (row + 2, col - 2),
                        (row + 1, col - 3),
                        (row, col - 3),
                        (row - 1, col - 3),
                        (row - 2, col - 2),
                        (row - 3, col - 1),
                    ];

                    for &(y, x) in &circle {
                        let val = image.at(y, x)?[0];
                        if i32::from(val) > i32::from(center) + self.threshold {
                            brighter += 1;
                        } else if i32::from(val) < i32::from(center) - self.threshold {
                            darker += 1;
                        }
                    }

                    if brighter >= 9 || darker >= 9 {
                        // Compute corner score
                        let score = self.compute_corner_score(image, row, col, center)?;

                        if score > self.threshold as f32 {
                            let physical_scale = 1 << level.octave;
                            let kp = KeyPoint {
                                pt: Point::new(col as i32 * physical_scale, row as i32 * physical_scale),
                                size: level.scale * self.pattern_scale,
                                angle: 0.0, // Will be computed in descriptor
                                response: score,
                                octave: level.octave as i32,
                            };
                            keypoints.push(kp);
                        }
                    }
                }
            }
        }

        Ok(keypoints)
    }

    fn compute_corner_score(&self, image: &Mat, row: usize, col: usize, center: u8) -> Result<f32> {
        let mut max_score = 0.0f32;

        let circle = [
            (row - 3, col),
            (row - 3, col + 1),
            (row - 2, col + 2),
            (row - 1, col + 3),
            (row, col + 3),
            (row + 1, col + 3),
            (row + 2, col + 2),
            (row + 3, col + 1),
            (row + 3, col),
            (row + 3, col - 1),
            (row + 2, col - 2),
            (row + 1, col - 3),
            (row, col - 3),
            (row - 1, col - 3),
            (row - 2, col - 2),
            (row - 3, col - 1),
        ];

        for &(y, x) in &circle {
            let val = image.at(y, x)?[0];
            let diff = (i32::from(val) - i32::from(center)).abs() as f32;
            max_score = max_score.max(diff);
        }

        Ok(max_score)
    }

    /// Compute BRISK descriptors using sampling pattern
    fn compute_descriptors(&self, image: &Mat, keypoints: &[KeyPoint]) -> Result<Vec<Vec<u8>>> {
        let mut descriptors = Vec::new();

        // BRISK sampling pattern: concentric circles
        let pattern = self.generate_sampling_pattern();

        for kp in keypoints {
            let row = kp.pt.y as usize;
            let col = kp.pt.x as usize;

            if row < 20 || row >= image.rows() - 20 || col < 20 || col >= image.cols() - 20 {
                continue;
            }

            // Compute orientation
            let angle = self.compute_orientation(image, row, col, &pattern)?;

            // Compute descriptor with rotation
            let mut descriptor = vec![0u8; 64]; // 512 bits
            let mut bit_idx = 0;

            let cos_angle = angle.cos();
            let sin_angle = angle.sin();

            // Compare short-distance pairs
            for i in 0..pattern.short_pairs.len() {
                if bit_idx >= 512 {
                    break;
                }

                let (p1, p2) = pattern.short_pairs[i];

                // Rotate points
                let (y1, x1) = self.rotate_point(p1.0, p1.1, cos_angle, sin_angle);
                let (y2, x2) = self.rotate_point(p2.0, p2.1, cos_angle, sin_angle);

                let y1_abs = (row as i32 + y1).max(0).min(image.rows() as i32 - 1) as usize;
                let x1_abs = (col as i32 + x1).max(0).min(image.cols() as i32 - 1) as usize;
                let y2_abs = (row as i32 + y2).max(0).min(image.rows() as i32 - 1) as usize;
                let x2_abs = (col as i32 + x2).max(0).min(image.cols() as i32 - 1) as usize;

                let val1 = image.at(y1_abs, x1_abs)?[0];
                let val2 = image.at(y2_abs, x2_abs)?[0];

                if val1 > val2 {
                    descriptor[bit_idx / 8] |= 1 << (bit_idx % 8);
                }

                bit_idx += 1;
            }

            descriptors.push(descriptor);
        }

        Ok(descriptors)
    }

    fn compute_orientation(&self, image: &Mat, row: usize, col: usize, pattern: &BriskPattern) -> Result<f32> {
        let mut gx = 0.0f32;
        let mut gy = 0.0f32;

        // Use long-distance pairs for orientation
        for &(p1, p2) in &pattern.long_pairs {
            let y1 = (row as i32 + p1.0).max(0).min(image.rows() as i32 - 1) as usize;
            let x1 = (col as i32 + p1.1).max(0).min(image.cols() as i32 - 1) as usize;
            let y2 = (row as i32 + p2.0).max(0).min(image.rows() as i32 - 1) as usize;
            let x2 = (col as i32 + p2.1).max(0).min(image.cols() as i32 - 1) as usize;

            let val1 = f32::from(image.at(y1, x1)?[0]);
            let val2 = f32::from(image.at(y2, x2)?[0]);

            let diff = val1 - val2;
            gx += diff * (p2.1 - p1.1) as f32;
            gy += diff * (p2.0 - p1.0) as f32;
        }

        Ok(gy.atan2(gx))
    }

    fn rotate_point(&self, dy: i32, dx: i32, cos_angle: f32, sin_angle: f32) -> (i32, i32) {
        let y = (dy as f32 * cos_angle - dx as f32 * sin_angle) as i32;
        let x = (dy as f32 * sin_angle + dx as f32 * cos_angle) as i32;
        (y, x)
    }

    fn generate_sampling_pattern(&self) -> BriskPattern {
        let mut short_pairs = Vec::new();
        let mut long_pairs = Vec::new();

        // Generate points on concentric circles
        let radii = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
        let num_points = [6, 8, 10, 12, 14, 16];

        let mut points = Vec::new();

        for (radius_idx, &radius) in radii.iter().enumerate() {
            let n_points = num_points[radius_idx];
            for i in 0..n_points {
                let angle = 2.0 * PI as f32 * i as f32 / n_points as f32;
                let scaled_radius = radius * self.pattern_scale;
                let dy = (scaled_radius * angle.sin()) as i32;
                let dx = (scaled_radius * angle.cos()) as i32;
                points.push((dy, dx));
            }
        }

        // Generate pairs
        let short_threshold = 6.0 * self.pattern_scale;
        let long_threshold = 10.0 * self.pattern_scale;

        for i in 0..points.len() {
            for j in i + 1..points.len() {
                let (dy1, dx1) = points[i];
                let (dy2, dx2) = points[j];

                let dist = (((dy2 - dy1) * (dy2 - dy1) + (dx2 - dx1) * (dx2 - dx1)) as f32).sqrt();

                if dist < short_threshold {
                    short_pairs.push((points[i], points[j]));
                    if short_pairs.len() >= 512 {
                        break;
                    }
                } else if dist > long_threshold {
                    long_pairs.push((points[i], points[j]));
                }
            }
            if short_pairs.len() >= 512 {
                break;
            }
        }

        // Ensure we have exactly 512 short pairs
        while short_pairs.len() < 512 {
            short_pairs.push(((0, 0), (1, 0)));
        }
        short_pairs.truncate(512);

        BriskPattern {
            short_pairs,
            long_pairs,
        }
    }
}

struct ScaleLevel {
    image: Mat,
    scale: f32,
    octave: usize,
    layer: usize,
}

struct BriskPattern {
    short_pairs: Vec<((i32, i32), (i32, i32))>,  // For descriptor
    long_pairs: Vec<((i32, i32), (i32, i32))>,   // For orientation
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_brisk() {
        let img = Mat::new_with_default(256, 256, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let brisk = BRISK::new(30, 3);
        let (keypoints, descriptors) = brisk.detect_and_compute(&img).unwrap();

        assert_eq!(keypoints.len(), descriptors.len());
        if !descriptors.is_empty() {
            assert_eq!(descriptors[0].len(), 64); // 512 bits = 64 bytes
        }
    }

    #[test]
    fn test_pattern_generation() {
        let brisk = BRISK::new(30, 3);
        let pattern = brisk.generate_sampling_pattern();
        assert_eq!(pattern.short_pairs.len(), 512);
        assert!(!pattern.long_pairs.is_empty());
    }

    #[test]
    fn test_corner_score() {
        let img = Mat::new_with_default(64, 64, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let brisk = BRISK::new(30, 3);
        let score = brisk.compute_corner_score(&img, 32, 32, 128).unwrap();
        assert!(score >= 0.0);
    }
}
