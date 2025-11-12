use crate::core::Mat;
use crate::core::types::Point;
use crate::error::{Error, Result};
use crate::features2d::KeyPoint;

/// BRIEF (Binary Robust Independent Elementary Features) descriptor
pub struct BRIEF {
    pub bytes: usize,
    pub patch_size: i32,
    pub use_orientation: bool,
    test_pairs: Vec<(Point, Point)>,
}

impl Default for BRIEF {
    fn default() -> Self {
        Self::new()
    }
}

impl BRIEF {
    /// Create BRIEF descriptor with default parameters
    #[must_use] 
    pub fn new() -> Self {
        Self::with_params(32, 48, false)
    }

    /// Create BRIEF descriptor with custom parameters
    #[must_use] 
    pub fn with_params(bytes: usize, patch_size: i32, use_orientation: bool) -> Self {
        let mut brief = Self {
            bytes,
            patch_size,
            use_orientation,
            test_pairs: Vec::new(),
        };

        // Generate test pairs using Gaussian distribution
        brief.generate_test_pairs();
        brief
    }

    fn generate_test_pairs(&mut self) {
        

        // Simple pseudo-random generator for reproducibility
        let mut rng = SimpleRng::new(42);

        let sigma = f64::from(self.patch_size) / 5.0;
        let num_pairs = self.bytes * 8;

        for _ in 0..num_pairs {
            // Generate points using Gaussian distribution around center
            let x1 = rng.gaussian(0.0, sigma);
            let y1 = rng.gaussian(0.0, sigma);
            let x2 = rng.gaussian(0.0, sigma);
            let y2 = rng.gaussian(0.0, sigma);

            let p1 = Point::new(x1 as i32, y1 as i32);
            let p2 = Point::new(x2 as i32, y2 as i32);

            self.test_pairs.push((p1, p2));
        }
    }

    /// Compute BRIEF descriptors for keypoints
    pub fn compute(&self, image: &Mat, keypoints: &[KeyPoint]) -> Result<Vec<Vec<u8>>> {
        if image.channels() != 1 {
            return Err(Error::InvalidParameter(
                "BRIEF requires grayscale image".to_string(),
            ));
        }

        let mut descriptors = Vec::new();

        for kp in keypoints {
            let descriptor = self.compute_descriptor(image, kp)?;
            descriptors.push(descriptor);
        }

        Ok(descriptors)
    }

    fn compute_descriptor(&self, image: &Mat, keypoint: &KeyPoint) -> Result<Vec<u8>> {
        let mut descriptor = vec![0u8; self.bytes];

        let center_x = keypoint.pt.x;
        let center_y = keypoint.pt.y;

        // Compute rotation matrix if using orientation
        let (cos_angle, sin_angle) = if self.use_orientation {
            let angle = f64::from(keypoint.angle) * std::f64::consts::PI / 180.0;
            (angle.cos(), angle.sin())
        } else {
            (1.0, 0.0)
        };

        for byte_idx in 0..self.bytes {
            let mut byte_value = 0u8;

            for bit_idx in 0..8 {
                let pair_idx = byte_idx * 8 + bit_idx;
                if pair_idx >= self.test_pairs.len() {
                    break;
                }

                let (p1, p2) = &self.test_pairs[pair_idx];

                // Apply rotation if needed
                let x1 = if self.use_orientation {
                    center_x + (f64::from(p1.x) * cos_angle - f64::from(p1.y) * sin_angle) as i32
                } else {
                    center_x + p1.x
                };

                let y1 = if self.use_orientation {
                    center_y + (f64::from(p1.x) * sin_angle + f64::from(p1.y) * cos_angle) as i32
                } else {
                    center_y + p1.y
                };

                let x2 = if self.use_orientation {
                    center_x + (f64::from(p2.x) * cos_angle - f64::from(p2.y) * sin_angle) as i32
                } else {
                    center_x + p2.x
                };

                let y2 = if self.use_orientation {
                    center_y + (f64::from(p2.x) * sin_angle + f64::from(p2.y) * cos_angle) as i32
                } else {
                    center_y + p2.y
                };

                // Clamp coordinates
                let x1 = x1.max(0).min(image.cols() as i32 - 1) as usize;
                let y1 = y1.max(0).min(image.rows() as i32 - 1) as usize;
                let x2 = x2.max(0).min(image.cols() as i32 - 1) as usize;
                let y2 = y2.max(0).min(image.rows() as i32 - 1) as usize;

                // Compare intensities
                let intensity1 = image.at(y1, x1)?[0];
                let intensity2 = image.at(y2, x2)?[0];

                if intensity1 < intensity2 {
                    byte_value |= 1 << bit_idx;
                }
            }

            descriptor[byte_idx] = byte_value;
        }

        Ok(descriptor)
    }
}

/// Simple pseudo-random number generator
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> f64 {
        // Linear congruential generator
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.state / 65536) % 32768) as f64 / 32768.0
    }

    fn gaussian(&mut self, mean: f64, std_dev: f64) -> f64 {
        // Box-Muller transform
        let u1 = self.next();
        let u2 = self.next();

        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std_dev * z0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{MatDepth, types::{Scalar, Point}};

    #[test]
    fn test_brief_descriptor() {
        let image = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let mut kp1 = KeyPoint::new(Point::new(50, 50), 10.0);
        kp1.angle = 0.0;

        let mut kp2 = KeyPoint::new(Point::new(30, 30), 10.0);
        kp2.angle = 45.0;

        let keypoints = vec![kp1, kp2];

        let brief = BRIEF::new();
        let descriptors = brief.compute(&image, &keypoints).unwrap();

        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0].len(), 32);
    }

    #[test]
    fn test_brief_with_orientation() {
        let image = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let mut kp = KeyPoint::new(Point::new(50, 50), 10.0);
        kp.angle = 90.0;

        let keypoints = vec![kp];

        let brief = BRIEF::with_params(32, 48, true);
        let descriptors = brief.compute(&image, &keypoints).unwrap();

        assert_eq!(descriptors.len(), 1);
        assert_eq!(descriptors[0].len(), 32);
    }
}
