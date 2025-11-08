use crate::core::{Mat, MatDepth};
use crate::core::types::Point2f;
use crate::error::{Error, Result};
use crate::features2d::KeyPoint;

/// FREAK (Fast Retina Keypoint) descriptor
/// Inspired by the human visual system's retinal pattern
pub struct FREAK {
    pub orientation_normalized: bool,
    pub scale_normalized: bool,
    pub pattern_scale: f32,
    receptive_fields: Vec<ReceptiveField>,
    pairs: Vec<(usize, usize)>,
}

#[derive(Clone)]
struct ReceptiveField {
    center: Point2f,
    radius: f32,
    layer: usize,
}

impl FREAK {
    /// Create FREAK descriptor with default parameters
    pub fn new() -> Self {
        Self::with_params(true, true, 22.0)
    }

    /// Create FREAK descriptor with custom parameters
    pub fn with_params(
        orientation_normalized: bool,
        scale_normalized: bool,
        pattern_scale: f32,
    ) -> Self {
        let mut freak = Self {
            orientation_normalized,
            scale_normalized,
            pattern_scale,
            receptive_fields: Vec::new(),
            pairs: Vec::new(),
        };

        freak.build_pattern();
        freak.build_pairs();
        freak
    }

    fn build_pattern(&mut self) {
        // Build retinal pattern with multiple layers
        // Inner layers have smaller receptive fields, outer layers have larger ones

        let num_layers = 6;
        let num_points_per_layer = [1, 6, 6, 6, 6, 6];

        for layer in 0..num_layers {
            let radius_scale = 2.0f32.powi(layer as i32);
            let base_radius = self.pattern_scale / 20.0 * radius_scale;

            if layer == 0 {
                // Center point
                self.receptive_fields.push(ReceptiveField {
                    center: Point2f::new(0.0, 0.0),
                    radius: base_radius,
                    layer,
                });
            } else {
                // Points arranged in a circle
                let num_points = num_points_per_layer[layer];
                let circle_radius = self.pattern_scale / 5.0 * radius_scale;

                for i in 0..num_points {
                    let angle = 2.0 * std::f32::consts::PI * i as f32 / num_points as f32;
                    let x = circle_radius * angle.cos();
                    let y = circle_radius * angle.sin();

                    self.receptive_fields.push(ReceptiveField {
                        center: Point2f::new(x, y),
                        radius: base_radius,
                        layer,
                    });
                }
            }
        }
    }

    fn build_pairs(&mut self) {
        // Build pairs for binary tests
        // Use pairs that maximize discrimination while being rotation-invariant

        let num_pairs = 512; // 64 bytes * 8 bits

        // Simple pairing strategy: all unique pairs
        for i in 0..self.receptive_fields.len() {
            for j in i + 1..self.receptive_fields.len() {
                if self.pairs.len() >= num_pairs {
                    return;
                }

                self.pairs.push((i, j));
            }
        }

        // If we don't have enough unique pairs, repeat some
        while self.pairs.len() < num_pairs {
            let idx = self.pairs.len() % (self.receptive_fields.len().saturating_sub(1).max(1));
            let i = idx;
            let j = (idx + 1) % self.receptive_fields.len();
            self.pairs.push((i, j));
        }
    }

    /// Compute FREAK descriptors for keypoints
    pub fn compute(&self, image: &Mat, keypoints: &[KeyPoint]) -> Result<Vec<Vec<u8>>> {
        if image.channels() != 1 {
            return Err(Error::InvalidParameter(
                "FREAK requires grayscale image".to_string(),
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
        let descriptor_bytes = 64; // 512 bits
        let mut descriptor = vec![0u8; descriptor_bytes];

        // Compute smoothed intensities for all receptive fields
        let intensities = self.compute_receptive_field_intensities(image, keypoint)?;

        // Compute orientation if needed
        let angle = if self.orientation_normalized {
            self.compute_orientation(&intensities)
        } else {
            keypoint.angle as f32
        };

        let cos_angle = (angle * std::f32::consts::PI / 180.0).cos();
        let sin_angle = (angle * std::f32::consts::PI / 180.0).sin();

        // Build binary descriptor from pairs
        for byte_idx in 0..descriptor_bytes {
            let mut byte_value = 0u8;

            for bit_idx in 0..8 {
                let pair_idx = byte_idx * 8 + bit_idx;
                if pair_idx >= self.pairs.len() {
                    break;
                }

                let (idx1, idx2) = self.pairs[pair_idx];

                // Compare intensities
                if intensities[idx1] < intensities[idx2] {
                    byte_value |= 1 << bit_idx;
                }
            }

            descriptor[byte_idx] = byte_value;
        }

        Ok(descriptor)
    }

    fn compute_receptive_field_intensities(
        &self,
        image: &Mat,
        keypoint: &KeyPoint,
    ) -> Result<Vec<f32>> {
        let mut intensities = Vec::new();

        let scale = if self.scale_normalized {
            keypoint.size / 10.0
        } else {
            1.0
        };

        for rf in &self.receptive_fields {
            // Transform receptive field center based on keypoint
            let x = keypoint.pt.x as f32 + rf.center.x * scale;
            let y = keypoint.pt.y as f32 + rf.center.y * scale;

            // Sample in a small region around the receptive field
            let mut sum = 0.0f32;
            let mut count = 0;

            let sample_radius = (rf.radius * scale).max(1.0) as i32;

            for dy in -sample_radius..=sample_radius {
                for dx in -sample_radius..=sample_radius {
                    let px = (x + dx as f32) as i32;
                    let py = (y + dy as f32) as i32;

                    if px >= 0 && px < image.cols() as i32 && py >= 0 && py < image.rows() as i32 {
                        let pixel = image.at(py as usize, px as usize)?;
                        sum += pixel[0] as f32;
                        count += 1;
                    }
                }
            }

            let intensity = if count > 0 {
                sum / count as f32
            } else {
                0.0
            };

            intensities.push(intensity);
        }

        Ok(intensities)
    }

    fn compute_orientation(&self, intensities: &[f32]) -> f32 {
        // Compute dominant orientation using gradient moments
        let mut gx = 0.0f32;
        let mut gy = 0.0f32;

        for i in 0..self.receptive_fields.len().min(intensities.len()) {
            let rf = &self.receptive_fields[i];
            let intensity = intensities[i];

            // Use position-weighted gradients
            gx += rf.center.x * intensity;
            gy += rf.center.y * intensity;
        }

        gy.atan2(gx) * 180.0 / std::f32::consts::PI
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{Scalar, Point};

    #[test]
    fn test_freak_descriptor() {
        let image = Mat::new_with_default(200, 200, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let mut kp1 = KeyPoint::new(Point::new(100, 100), 20.0);
        kp1.angle = 0.0;

        let mut kp2 = KeyPoint::new(Point::new(60, 60), 15.0);
        kp2.angle = 45.0;

        let keypoints = vec![kp1, kp2];

        let freak = FREAK::new();
        let descriptors = freak.compute(&image, &keypoints).unwrap();

        assert_eq!(descriptors.len(), 2);
        assert_eq!(descriptors[0].len(), 64);
    }

    #[test]
    fn test_freak_pattern() {
        let freak = FREAK::new();

        // Should have created receptive fields
        assert!(freak.receptive_fields.len() > 0);

        // Should have created pairs
        assert_eq!(freak.pairs.len(), 512);
    }

    #[test]
    fn test_freak_without_normalization() {
        let image = Mat::new_with_default(200, 200, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let mut kp = KeyPoint::new(Point::new(100, 100), 20.0);
        kp.angle = 0.0;

        let keypoints = vec![kp];

        let freak = FREAK::with_params(false, false, 22.0);
        let descriptors = freak.compute(&image, &keypoints).unwrap();

        assert_eq!(descriptors.len(), 1);
        assert_eq!(descriptors[0].len(), 64);
    }
}
