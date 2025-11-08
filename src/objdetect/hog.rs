use crate::core::{Mat, MatDepth};
use crate::core::types::{Size, Rect};
use crate::error::{Error, Result};
use std::f64::consts::PI;

/// HOG (Histogram of Oriented Gradients) Descriptor
pub struct HOGDescriptor {
    pub win_size: Size,
    pub block_size: Size,
    pub block_stride: Size,
    pub cell_size: Size,
    pub nbins: usize,
}

impl HOGDescriptor {
    pub fn new() -> Self {
        Self {
            win_size: Size::new(64, 128),
            block_size: Size::new(16, 16),
            block_stride: Size::new(8, 8),
            cell_size: Size::new(8, 8),
            nbins: 9,
        }
    }

    /// Compute HOG descriptor for an image
    pub fn compute(&self, img: &Mat) -> Result<Vec<f32>> {
        if img.channels() != 1 {
            return Err(Error::InvalidParameter(
                "HOG requires grayscale image".to_string(),
            ));
        }

        // Calculate gradients
        use crate::imgproc::sobel;

        let mut grad_x = Mat::new(1, 1, 1, MatDepth::U8)?;
        let mut grad_y = Mat::new(1, 1, 1, MatDepth::U8)?;

        sobel(img, &mut grad_x, 1, 0, 3)?;
        sobel(img, &mut grad_y, 0, 1, 3)?;

        // Calculate magnitude and orientation
        let mut magnitudes = vec![vec![0.0f32; img.cols()]; img.rows()];
        let mut orientations = vec![vec![0.0f32; img.cols()]; img.rows()];

        for row in 0..img.rows() {
            for col in 0..img.cols() {
                let gx = grad_x.at(row, col)?[0] as f32;
                let gy = grad_y.at(row, col)?[0] as f32;

                magnitudes[row][col] = (gx * gx + gy * gy).sqrt();
                orientations[row][col] = gy.atan2(gx);
            }
        }

        // Compute cell histograms
        let cells_per_block_x = self.block_size.width / self.cell_size.width;
        let cells_per_block_y = self.block_size.height / self.cell_size.height;

        let mut descriptors = Vec::new();

        // Iterate over blocks
        let num_blocks_y = (img.rows() as i32 - self.block_size.height) / self.block_stride.height + 1;
        let num_blocks_x = (img.cols() as i32 - self.block_size.width) / self.block_stride.width + 1;

        for block_y in 0..num_blocks_y {
            for block_x in 0..num_blocks_x {
                let block_start_x = block_x * self.block_stride.width;
                let block_start_y = block_y * self.block_stride.height;

                let mut block_descriptor = Vec::new();

                // Compute histogram for each cell in the block
                for cell_y in 0..cells_per_block_y {
                    for cell_x in 0..cells_per_block_x {
                        let cell_start_x = block_start_x + cell_x * self.cell_size.width;
                        let cell_start_y = block_start_y + cell_y * self.cell_size.height;

                        let hist = self.compute_cell_histogram(
                            &magnitudes,
                            &orientations,
                            cell_start_x as usize,
                            cell_start_y as usize,
                        );

                        block_descriptor.extend(hist);
                    }
                }

                // Normalize block descriptor
                let norm: f32 = block_descriptor.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for val in &mut block_descriptor {
                        *val /= norm;
                    }
                }

                descriptors.extend(block_descriptor);
            }
        }

        Ok(descriptors)
    }

    fn compute_cell_histogram(
        &self,
        magnitudes: &[Vec<f32>],
        orientations: &[Vec<f32>],
        start_x: usize,
        start_y: usize,
    ) -> Vec<f32> {
        let mut histogram = vec![0.0f32; self.nbins];
        let angle_per_bin = PI as f32 / self.nbins as f32;

        for y in start_y..(start_y + self.cell_size.height as usize) {
            for x in start_x..(start_x + self.cell_size.width as usize) {
                if y < magnitudes.len() && x < magnitudes[0].len() {
                    let mag = magnitudes[y][x];
                    let mut angle = orientations[y][x];

                    // Convert to [0, PI]
                    if angle < 0.0 {
                        angle += PI as f32;
                    }

                    // Find bin
                    let bin_idx = (angle / angle_per_bin) as usize;
                    let bin_idx = bin_idx.min(self.nbins - 1);

                    histogram[bin_idx] += mag;
                }
            }
        }

        histogram
    }

    /// Detect objects using HOG descriptor and SVM
    pub fn detect_multi_scale(
        &self,
        img: &Mat,
        hit_threshold: f64,
        win_stride: Size,
        scale: f64,
    ) -> Result<Vec<Rect>> {
        let mut detections = Vec::new();

        // Slide window across image at multiple scales
        let mut current_scale = 1.0;

        while current_scale < 3.0 {
            let stride_x = win_stride.width;
            let stride_y = win_stride.height;

            for y in (0..img.rows() as i32).step_by(stride_y as usize) {
                for x in (0..img.cols() as i32).step_by(stride_x as usize) {
                    if x + self.win_size.width <= img.cols() as i32
                        && y + self.win_size.height <= img.rows() as i32
                    {
                        let window = Rect::new(x, y, self.win_size.width, self.win_size.height);

                        // Extract window and compute HOG
                        let roi = img.roi(window)?;
                        let descriptor = self.compute(&roi)?;

                        // Simple detection: check if descriptor energy is above threshold
                        let energy: f32 = descriptor.iter().sum();

                        if energy as f64 > hit_threshold {
                            detections.push(window);
                        }
                    }
                }
            }

            current_scale *= scale;
        }

        Ok(detections)
    }
}

impl Default for HOGDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_hog_descriptor() {
        let img = Mat::new_with_default(128, 64, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let hog = HOGDescriptor::new();
        let descriptor = hog.compute(&img).unwrap();

        assert!(!descriptor.is_empty());
    }
}
