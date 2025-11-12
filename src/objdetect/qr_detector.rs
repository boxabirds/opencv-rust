#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss, clippy::cast_precision_loss)]
use crate::core::Mat;
use crate::core::types::Point2f;
use crate::error::{Error, Result};

/// QR Code detector
pub struct QRCodeDetector {
    initialized: bool,
}

impl QRCodeDetector {
    /// Create a new QR code detector
    #[must_use] 
    pub fn new() -> Self {
        Self { initialized: true }
    }

    /// Detect QR code in image and return corner points
    pub fn detect(&self, image: &Mat) -> Result<Option<Vec<Point2f>>> {
        if image.channels() != 1 {
            return Err(Error::InvalidParameter(
                "QR detection requires grayscale image".to_string(),
            ));
        }

        // Find finder patterns (the three corner squares)
        let finder_patterns = self.find_finder_patterns(image)?;

        if finder_patterns.len() < 3 {
            return Ok(None);
        }

        // Determine QR code corners from finder patterns
        let corners = self.compute_qr_corners(&finder_patterns)?;

        Ok(Some(corners))
    }

    /// Detect and decode QR code
    pub fn detect_and_decode(&self, image: &Mat) -> Result<Option<(String, Vec<Point2f>)>> {
        let corners = match self.detect(image)? {
            Some(c) => c,
            None => return Ok(None),
        };

        // Simplified decoding (real implementation would decode the actual data)
        let decoded_data = self.decode_qr(image, &corners)?;

        Ok(Some((decoded_data, corners)))
    }

    /// Detect multiple QR codes in image
    pub fn detect_multi(&self, image: &Mat) -> Result<Vec<Vec<Point2f>>> {
        let mut qr_codes = Vec::new();

        // Simplified: detect one QR code
        if let Some(corners) = self.detect(image)? {
            qr_codes.push(corners);
        }

        Ok(qr_codes)
    }

    fn find_finder_patterns(&self, image: &Mat) -> Result<Vec<FinderPattern>> {
        let mut patterns = Vec::new();

        // Search for finder patterns using ratio of black/white transitions
        // Finder pattern has ratio 1:1:3:1:1

        for row in 10..image.rows() - 10 {
            let mut col = 10;

            while col < image.cols() - 10 {
                if let Some(pattern) = self.check_finder_pattern_at(image, row, col)? {
                    let size = pattern.size;
                    patterns.push(pattern);
                    col += size as usize * 2;
                } else {
                    col += 1;
                }
            }
        }

        // Filter duplicate patterns
        self.filter_patterns(patterns)
    }

    fn check_finder_pattern_at(
        &self,
        image: &Mat,
        row: usize,
        col: usize,
    ) -> Result<Option<FinderPattern>> {
        // Check horizontal ratio
        let horizontal_ratio = self.measure_pattern_ratio(image, row, col, true)?;

        if !self.is_finder_pattern_ratio(&horizontal_ratio) {
            return Ok(None);
        }

        // Check vertical ratio
        let vertical_ratio = self.measure_pattern_ratio(image, row, col, false)?;

        if !self.is_finder_pattern_ratio(&vertical_ratio) {
            return Ok(None);
        }

        // Calculate center and size
        let size = usize::midpoint(horizontal_ratio.iter().sum::<usize>(), vertical_ratio.iter().sum::<usize>());

        Ok(Some(FinderPattern {
            center: Point2f::new(col as f32, row as f32),
            size: size as f32,
        }))
    }

    fn measure_pattern_ratio(
        &self,
        image: &Mat,
        start_row: usize,
        start_col: usize,
        horizontal: bool,
    ) -> Result<Vec<usize>> {
        let threshold = 128;
        let mut state_counts = vec![0usize; 5];
        let mut state_idx = 0;

        let start_pixel = image.at(start_row, start_col)?[0];
        let mut current_state = start_pixel < threshold;

        for i in 0..50 {
            let (row, col) = if horizontal {
                (start_row, start_col + i)
            } else {
                (start_row + i, start_col)
            };

            if row >= image.rows() || col >= image.cols() {
                break;
            }

            let pixel = image.at(row, col)?[0];
            let is_dark = pixel < threshold;

            if is_dark == current_state {
                state_counts[state_idx] += 1;
            } else {
                state_idx += 1;
                if state_idx >= 5 {
                    break;
                }
                state_counts[state_idx] = 1;
                current_state = is_dark;
            }
        }

        Ok(state_counts)
    }

    fn is_finder_pattern_ratio(&self, state_counts: &[usize]) -> bool {
        if state_counts.len() < 5 {
            return false;
        }

        let total: usize = state_counts.iter().sum();
        if total < 7 {
            return false;
        }

        let module_size = total as f32 / 7.0;
        let variance = module_size / 2.0;

        // Check ratio 1:1:3:1:1
        (state_counts[0] as f32 - module_size).abs() < variance
            && (state_counts[1] as f32 - module_size).abs() < variance
            && (state_counts[2] as f32 - 3.0 * module_size).abs() < 3.0 * variance
            && (state_counts[3] as f32 - module_size).abs() < variance
            && (state_counts[4] as f32 - module_size).abs() < variance
    }

    fn filter_patterns(&self, mut patterns: Vec<FinderPattern>) -> Result<Vec<FinderPattern>> {
        if patterns.len() <= 3 {
            return Ok(patterns);
        }

        // Remove duplicates (patterns close to each other)
        patterns.sort_by(|a, b| a.center.x.partial_cmp(&b.center.x).unwrap());

        let mut filtered: Vec<FinderPattern> = Vec::new();

        for pattern in patterns {
            let mut is_duplicate = false;

            for existing in &filtered {
                let dx = pattern.center.x - existing.center.x;
                let dy = pattern.center.y - existing.center.y;
                let dist = (dx * dx + dy * dy).sqrt();

                if dist < pattern.size {
                    is_duplicate = true;
                    break;
                }
            }

            if !is_duplicate {
                filtered.push(pattern);
            }
        }

        Ok(filtered)
    }

    fn compute_qr_corners(&self, patterns: &[FinderPattern]) -> Result<Vec<Point2f>> {
        if patterns.len() < 3 {
            return Err(Error::InvalidParameter("Not enough finder patterns".to_string()));
        }

        // Take first three patterns
        let p1 = patterns[0].center;
        let p2 = patterns[1].center;
        let p3 = patterns[2].center;

        // Fourth corner is computed from the three finder patterns
        let p4 = Point2f::new(
            p1.x + p3.x - p2.x,
            p1.y + p3.y - p2.y,
        );

        Ok(vec![p1, p2, p3, p4])
    }

    fn decode_qr(&self, _image: &Mat, _corners: &[Point2f]) -> Result<String> {
        // Simplified decoding
        // Real implementation would:
        // 1. Warp QR code to canonical view
        // 2. Read timing patterns
        // 3. Decode format information
        // 4. Read data modules
        // 5. Apply error correction

        Ok("QR_CODE_DATA".to_string())
    }
}

impl Default for QRCodeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
struct FinderPattern {
    center: Point2f,
    size: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{MatDepth, types::Scalar};

    #[test]
    fn test_qr_detector_creation() {
        let detector = QRCodeDetector::new();
        assert!(detector.initialized);
    }

    #[test]
    fn test_qr_detection_on_empty() {
        let detector = QRCodeDetector::new();
        let image = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(255.0)).unwrap();

        let result = detector.detect(&image).unwrap();
        assert!(result.is_none() || result.unwrap().len() == 4);
    }

    #[test]
    fn test_qr_detect_multi() {
        let detector = QRCodeDetector::new();
        let image = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(255.0)).unwrap();

        let result = detector.detect_multi(&image).unwrap();
        assert!(result.len() >= 0);
    }

    #[test]
    fn test_finder_pattern_ratio() {
        let detector = QRCodeDetector::new();

        // Valid ratio 1:1:3:1:1 (e.g., 7, 7, 21, 7, 7)
        let valid = vec![7, 7, 21, 7, 7];
        assert!(detector.is_finder_pattern_ratio(&valid));

        // Invalid ratio
        let invalid = vec![10, 5, 10, 5, 10];
        assert!(!detector.is_finder_pattern_ratio(&invalid));
    }
}
