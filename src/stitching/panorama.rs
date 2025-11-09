use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};
use crate::features2d::KeyPoint;
use crate::core::types::Point;

/// Panorama stitcher for creating panoramic images from multiple images
pub struct PanoramaStitcher {
    confidence_threshold: f32,
    blend_strength: f32,
    pub warper_type: WarpType,
}

#[derive(Clone, Copy)]
pub enum WarpType {
    Cylindrical,
    Spherical,
    Plane,
}

impl PanoramaStitcher {
    pub fn new() -> Self {
        Self {
            confidence_threshold: 1.0,
            blend_strength: 5.0,
            warper_type: WarpType::Cylindrical,
        }
    }

    pub fn with_confidence(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    pub fn with_warp_type(mut self, warp_type: WarpType) -> Self {
        self.warper_type = warp_type;
        self
    }

    /// Stitch multiple images into a panorama
    pub fn stitch(&self, images: &[Mat]) -> Result<Mat> {
        if images.len() < 2 {
            return Err(Error::InvalidParameter(
                "Need at least 2 images for stitching".to_string(),
            ));
        }

        // 1. Find features in all images
        let mut all_keypoints = Vec::new();
        let mut all_descriptors = Vec::new();

        for img in images {
            let (kps, descs) = self.extract_features(img)?;
            all_keypoints.push(kps);
            all_descriptors.push(descs);
        }

        // 2. Match features between adjacent images
        let matches = self.match_images(&all_descriptors)?;

        // 3. Estimate homographies
        let homographies = self.estimate_homographies(&all_keypoints, &matches)?;

        // 4. Warp images to common coordinate frame
        let warped_images = self.warp_images(images, &homographies)?;

        // 5. Find optimal seams
        let seams = self.find_seams(&warped_images)?;

        // 6. Blend images
        let panorama = self.blend_images(&warped_images, &seams)?;

        Ok(panorama)
    }

    fn extract_features(&self, image: &Mat) -> Result<(Vec<KeyPoint>, Vec<Vec<u8>>)> {
        // Convert to grayscale if needed
        let gray = if image.channels() == 3 {
            self.rgb_to_gray(image)?
        } else {
            image.clone_mat()
        };

        // Use AKAZE features (simplified - would normally use actual AKAZE)
        let mut keypoints = Vec::new();
        let mut descriptors = Vec::new();

        // Sample keypoints across the image
        let step = 20;
        for row in (step..gray.rows() - step).step_by(step) {
            for col in (step..gray.cols() - step).step_by(step) {
                keypoints.push(KeyPoint {
                    pt: Point::new(col as i32, row as i32),
                    size: 5.0,
                    angle: 0.0,
                    response: 1.0,
                    octave: 0,
                });

                // Simple descriptor: local patch
                let mut desc = vec![0u8; 64];
                let mut idx = 0;
                for dy in -2..=2 {
                    for dx in -2..=2 {
                        if idx < 64 {
                            let y = (row as i32 + dy).clamp(0, gray.rows() as i32 - 1) as usize;
                            let x = (col as i32 + dx).clamp(0, gray.cols() as i32 - 1) as usize;
                            desc[idx] = gray.at(y, x)?[0];
                            idx += 1;
                        }
                    }
                }
                descriptors.push(desc);
            }
        }

        Ok((keypoints, descriptors))
    }

    fn rgb_to_gray(&self, image: &Mat) -> Result<Mat> {
        let mut gray = Mat::new(image.rows(), image.cols(), 1, image.depth())?;

        for row in 0..image.rows() {
            for col in 0..image.cols() {
                let pixel = image.at(row, col)?;
                let gray_val = (0.299 * pixel[0] as f32
                              + 0.587 * pixel[1] as f32
                              + 0.114 * pixel[2] as f32) as u8;
                gray.at_mut(row, col)?[0] = gray_val;
            }
        }

        Ok(gray)
    }

    fn match_images(&self, all_descriptors: &[Vec<Vec<u8>>]) -> Result<Vec<Vec<(usize, usize)>>> {
        let mut matches = Vec::new();

        // Match consecutive image pairs
        for i in 0..all_descriptors.len() - 1 {
            let desc1 = &all_descriptors[i];
            let desc2 = &all_descriptors[i + 1];

            let mut pair_matches = Vec::new();

            // Simple nearest neighbor matching
            for (idx1, d1) in desc1.iter().enumerate() {
                let mut best_dist = u32::MAX;
                let mut best_idx = 0;

                for (idx2, d2) in desc2.iter().enumerate() {
                    let dist = hamming_distance(d1, d2);
                    if dist < best_dist {
                        best_dist = dist;
                        best_idx = idx2;
                    }
                }

                // Threshold matching
                if best_dist < 50 {
                    pair_matches.push((idx1, best_idx));
                }
            }

            matches.push(pair_matches);
        }

        Ok(matches)
    }

    fn estimate_homographies(
        &self,
        all_keypoints: &[Vec<KeyPoint>],
        matches: &[Vec<(usize, usize)>],
    ) -> Result<Vec<[[f64; 3]; 3]>> {
        let mut homographies = Vec::new();

        // Identity for first image
        homographies.push([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]);

        // Estimate homography for each subsequent image
        for (i, match_pairs) in matches.iter().enumerate() {
            if match_pairs.len() < 4 {
                // Not enough matches, use identity
                homographies.push([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]);
                continue;
            }

            let kps1 = &all_keypoints[i];
            let kps2 = &all_keypoints[i + 1];

            // Extract point correspondences
            let mut src_points = Vec::new();
            let mut dst_points = Vec::new();

            for &(idx1, idx2) in match_pairs {
                src_points.push(kps1[idx1].pt);
                dst_points.push(kps2[idx2].pt);
            }

            // Simplified homography estimation
            let h = self.estimate_simple_homography(&src_points, &dst_points)?;
            homographies.push(h);
        }

        Ok(homographies)
    }

    fn estimate_simple_homography(
        &self,
        src: &[Point],
        dst: &[Point],
    ) -> Result<[[f64; 3]; 3]> {
        // Simplified: just compute translation and scale
        let mut tx = 0.0;
        let mut ty = 0.0;
        let count = src.len().min(dst.len());

        for i in 0..count {
            tx += (dst[i].x - src[i].x) as f64;
            ty += (dst[i].y - src[i].y) as f64;
        }

        tx /= count as f64;
        ty /= count as f64;

        Ok([
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ])
    }

    fn warp_images(
        &self,
        images: &[Mat],
        homographies: &[[[f64; 3]; 3]],
    ) -> Result<Vec<Mat>> {
        let mut warped = Vec::new();

        // Determine output size
        let max_width = images.iter().map(|img| img.cols()).max().unwrap_or(0);
        let max_height = images.iter().map(|img| img.rows()).max().unwrap_or(0);

        let output_width = max_width * 2; // Rough estimate
        let output_height = max_height;

        for (img, h) in images.iter().zip(homographies.iter()) {
            let warped_img = self.warp_perspective(img, h, output_width, output_height)?;
            warped.push(warped_img);
        }

        Ok(warped)
    }

    fn warp_perspective(
        &self,
        src: &Mat,
        h: &[[f64; 3]; 3],
        dst_width: usize,
        dst_height: usize,
    ) -> Result<Mat> {
        let mut dst = Mat::new(dst_height, dst_width, src.channels(), src.depth())?;

        for row in 0..dst_height {
            for col in 0..dst_width {
                // Apply inverse homography
                let x = col as f64;
                let y = row as f64;

                let denom = h[2][0] * x + h[2][1] * y + h[2][2];
                if denom.abs() < 1e-8 {
                    continue;
                }

                let src_x = (h[0][0] * x + h[0][1] * y + h[0][2]) / denom;
                let src_y = (h[1][0] * x + h[1][1] * y + h[1][2]) / denom;

                let src_col = src_x as i32;
                let src_row = src_y as i32;

                if src_row >= 0 && src_row < src.rows() as i32
                    && src_col >= 0 && src_col < src.cols() as i32
                {
                    for ch in 0..src.channels() {
                        dst.at_mut(row, col)?[ch] = src.at(src_row as usize, src_col as usize)?[ch];
                    }
                }
            }
        }

        Ok(dst)
    }

    fn find_seams(&self, images: &[Mat]) -> Result<Vec<Vec<usize>>> {
        // Simplified: vertical seams at midpoints
        let mut seams = Vec::new();

        if !images.is_empty() {
            let width = images[0].cols();
            for i in 0..images.len() - 1 {
                let seam_col = width / 2 + i * 100;
                let mut seam = Vec::new();
                for row in 0..images[0].rows() {
                    seam.push(seam_col);
                }
                seams.push(seam);
            }
        }

        Ok(seams)
    }

    fn blend_images(&self, images: &[Mat], _seams: &[Vec<usize>]) -> Result<Mat> {
        if images.is_empty() {
            return Err(Error::InvalidParameter("No images to blend".to_string()));
        }

        // Simple alpha blending
        let mut panorama = images[0].clone_mat();

        for img in &images[1..] {
            panorama = self.alpha_blend(&panorama, img, 0.5)?;
        }

        Ok(panorama)
    }

    fn alpha_blend(&self, img1: &Mat, img2: &Mat, alpha: f32) -> Result<Mat> {
        let rows = img1.rows().min(img2.rows());
        let cols = img1.cols().min(img2.cols());
        let channels = img1.channels().min(img2.channels());

        let mut result = Mat::new(rows, cols, channels, img1.depth())?;

        for row in 0..rows {
            for col in 0..cols {
                for ch in 0..channels {
                    let val1 = img1.at(row, col)?[ch] as f32;
                    let val2 = img2.at(row, col)?[ch] as f32;
                    let blended = (alpha * val1 + (1.0 - alpha) * val2) as u8;
                    result.at_mut(row, col)?[ch] = blended;
                }
            }
        }

        Ok(result)
    }
}

fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    let mut dist = 0;
    for (byte_a, byte_b) in a.iter().zip(b.iter()) {
        let xor = byte_a ^ byte_b;
        dist += xor.count_ones();
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_panorama_stitcher_creation() {
        let stitcher = PanoramaStitcher::new()
            .with_confidence(1.5)
            .with_warp_type(WarpType::Cylindrical);

        assert!((stitcher.confidence_threshold - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_feature_extraction() {
        let img = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let stitcher = PanoramaStitcher::new();
        let (keypoints, descriptors) = stitcher.extract_features(&img).unwrap();

        assert!(!keypoints.is_empty());
        assert_eq!(keypoints.len(), descriptors.len());
    }

    #[test]
    fn test_rgb_to_gray() {
        let img = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let stitcher = PanoramaStitcher::new();
        let gray = stitcher.rgb_to_gray(&img).unwrap();

        assert_eq!(gray.channels(), 1);
        assert_eq!(gray.rows(), 50);
    }
}
