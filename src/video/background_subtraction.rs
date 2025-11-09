use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};

/// Mixture of Gaussians (MOG2) background subtractor
pub struct BackgroundSubtractorMOG2 {
    pub history: usize,
    pub var_threshold: f64,
    pub detect_shadows: bool,
    num_gaussians: usize,
    learning_rate: f64,
    background_ratio: f64,
    var_init: f64,
    var_min: f64,
    var_max: f64,

    // Model storage
    mean: Vec<Vec<Vec<f32>>>,  // [row][col][gaussian_idx]
    variance: Vec<Vec<Vec<f32>>>,
    weight: Vec<Vec<Vec<f32>>>,
    frame_count: usize,
}

impl BackgroundSubtractorMOG2 {
    /// Create MOG2 background subtractor with default parameters
    pub fn new() -> Self {
        Self::with_params(500, 16.0, true)
    }

    /// Create MOG2 background subtractor with custom parameters
    pub fn with_params(history: usize, var_threshold: f64, detect_shadows: bool) -> Self {
        Self {
            history,
            var_threshold,
            detect_shadows,
            num_gaussians: 5,
            learning_rate: -1.0, // Auto
            background_ratio: 0.9,
            var_init: 15.0,
            var_min: 4.0,
            var_max: 75.0,
            mean: Vec::new(),
            variance: Vec::new(),
            weight: Vec::new(),
            frame_count: 0,
        }
    }

    /// Apply background subtraction
    pub fn apply(&mut self, image: &Mat, fgmask: &mut Mat, learning_rate: f64) -> Result<()> {
        if image.channels() != 3 {
            return Err(Error::InvalidParameter(
                "MOG2 requires 3-channel image".to_string(),
            ));
        }

        let rows = image.rows();
        let cols = image.cols();

        // Initialize model if needed
        if self.mean.is_empty() {
            self.mean = vec![vec![vec![0.0; self.num_gaussians]; cols]; rows];
            self.variance = vec![vec![vec![self.var_init as f32; self.num_gaussians]; cols]; rows];
            self.weight = vec![vec![vec![0.0; self.num_gaussians]; cols]; rows];
        }

        *fgmask = Mat::new(rows, cols, 1, MatDepth::U8)?;

        let alpha = if learning_rate < 0.0 {
            1.0 / self.history as f64
        } else {
            learning_rate
        };

        self.frame_count += 1;

        for row in 0..rows {
            for col in 0..cols {
                let pixel = image.at(row, col)?;
                let intensity = (pixel[0] as f32 + pixel[1] as f32 + pixel[2] as f32) / 3.0;

                let mut matched = false;
                let mut match_idx = 0;

                // Check if pixel matches any Gaussian
                for k in 0..self.num_gaussians {
                    let diff = (intensity - self.mean[row][col][k]).abs();
                    let threshold = (self.var_threshold * self.variance[row][col][k].sqrt() as f64) as f32;

                    if diff < threshold {
                        matched = true;
                        match_idx = k;
                        break;
                    }
                }

                let mut is_background = false;

                if matched {
                    // Update matched Gaussian
                    let k = match_idx;
                    let rho = alpha * self.weight[row][col][k] as f64;

                    self.mean[row][col][k] =
                        ((1.0 - rho) * self.mean[row][col][k] as f64 + rho * intensity as f64) as f32;

                    let diff = intensity - self.mean[row][col][k];
                    self.variance[row][col][k] =
                        ((1.0 - rho) * self.variance[row][col][k] as f64 + rho * diff as f64 * diff as f64) as f32;

                    self.variance[row][col][k] = self.variance[row][col][k]
                        .max(self.var_min as f32)
                        .min(self.var_max as f32);

                    // Check if it's background
                    let mut weight_sum = 0.0f32;
                    for k in 0..self.num_gaussians {
                        weight_sum += self.weight[row][col][k];
                        if weight_sum > self.background_ratio as f32 {
                            if k >= match_idx {
                                is_background = true;
                            }
                            break;
                        }
                    }
                } else {
                    // Replace least probable Gaussian
                    let k = self.num_gaussians - 1;
                    self.mean[row][col][k] = intensity;
                    self.variance[row][col][k] = self.var_init as f32;
                    self.weight[row][col][k] = 0.05;
                }

                // Update weights
                for k in 0..self.num_gaussians {
                    if k == match_idx && matched {
                        self.weight[row][col][k] =
                            ((1.0 - alpha) * self.weight[row][col][k] as f64 + alpha) as f32;
                    } else {
                        self.weight[row][col][k] =
                            ((1.0 - alpha) * self.weight[row][col][k] as f64) as f32;
                    }
                }

                // Normalize weights
                let weight_sum: f32 = self.weight[row][col].iter().sum();
                if weight_sum > 0.0 {
                    for k in 0..self.num_gaussians {
                        self.weight[row][col][k] /= weight_sum;
                    }
                }

                // Sort Gaussians by weight/variance
                let mut indices: Vec<usize> = (0..self.num_gaussians).collect();
                indices.sort_by(|&a, &b| {
                    let score_a = self.weight[row][col][a] / self.variance[row][col][a].sqrt();
                    let score_b = self.weight[row][col][b] / self.variance[row][col][b].sqrt();
                    score_b.partial_cmp(&score_a).unwrap()
                });

                // Reorder
                let mean_copy = self.mean[row][col].clone();
                let var_copy = self.variance[row][col].clone();
                let weight_copy = self.weight[row][col].clone();

                for (new_idx, &old_idx) in indices.iter().enumerate() {
                    self.mean[row][col][new_idx] = mean_copy[old_idx];
                    self.variance[row][col][new_idx] = var_copy[old_idx];
                    self.weight[row][col][new_idx] = weight_copy[old_idx];
                }

                // Set foreground mask
                let fg_pixel = fgmask.at_mut(row, col)?;
                fg_pixel[0] = if is_background { 0 } else { 255 };
            }
        }

        Ok(())
    }

    /// Get background image
    pub fn get_background_image(&self, background: &mut Mat) -> Result<()> {
        if self.mean.is_empty() {
            return Err(Error::InvalidParameter(
                "Model not initialized".to_string(),
            ));
        }

        let rows = self.mean.len();
        let cols = self.mean[0].len();

        *background = Mat::new(rows, cols, 3, MatDepth::U8)?;

        for row in 0..rows {
            for col in 0..cols {
                // Use the most probable Gaussian (first after sorting)
                let intensity = self.mean[row][col][0];

                let bg_pixel = background.at_mut(row, col)?;
                bg_pixel[0] = intensity as u8;
                bg_pixel[1] = intensity as u8;
                bg_pixel[2] = intensity as u8;
            }
        }

        Ok(())
    }
}

/// K-Nearest Neighbors background subtractor
pub struct BackgroundSubtractorKNN {
    pub history: usize,
    pub dist2_threshold: f64,
    pub detect_shadows: bool,
    k_nn_samples: usize,

    // Sample storage
    samples: Vec<Vec<Vec<f32>>>,  // [row][col][sample_idx]
    sample_idx: Vec<Vec<usize>>,  // Current write position
    frame_count: usize,
}

impl BackgroundSubtractorKNN {
    /// Create KNN background subtractor with default parameters
    pub fn new() -> Self {
        Self::with_params(500, 400.0, true)
    }

    /// Create KNN background subtractor with custom parameters
    pub fn with_params(history: usize, dist2_threshold: f64, detect_shadows: bool) -> Self {
        Self {
            history,
            dist2_threshold,
            detect_shadows,
            k_nn_samples: 5,
            samples: Vec::new(),
            sample_idx: Vec::new(),
            frame_count: 0,
        }
    }

    /// Apply background subtraction
    pub fn apply(&mut self, image: &Mat, fgmask: &mut Mat, learning_rate: f64) -> Result<()> {
        if image.channels() != 3 {
            return Err(Error::InvalidParameter(
                "KNN requires 3-channel image".to_string(),
            ));
        }

        let rows = image.rows();
        let cols = image.cols();

        // Initialize samples if needed
        if self.samples.is_empty() {
            let max_samples = self.history.min(100);
            self.samples = vec![vec![vec![0.0; max_samples]; cols]; rows];
            self.sample_idx = vec![vec![0; cols]; rows];
        }

        *fgmask = Mat::new(rows, cols, 1, MatDepth::U8)?;

        self.frame_count += 1;

        let max_samples = self.samples[0][0].len();

        for row in 0..rows {
            for col in 0..cols {
                let pixel = image.at(row, col)?;
                let intensity = (pixel[0] as f32 + pixel[1] as f32 + pixel[2] as f32) / 3.0;

                // Find k-nearest samples
                let mut distances: Vec<f32> = Vec::new();

                let num_samples = self.sample_idx[row][col].min(max_samples);

                for i in 0..num_samples {
                    let sample = self.samples[row][col][i];
                    let diff = intensity - sample;
                    distances.push(diff * diff);
                }

                if distances.is_empty() {
                    distances.push(f32::MAX);
                }

                distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

                // Use k-nearest to determine foreground
                let k = self.k_nn_samples.min(distances.len());
                let avg_dist: f32 = distances.iter().take(k).sum::<f32>() / k as f32;

                let is_background = avg_dist < self.dist2_threshold as f32;

                // Update samples
                // learning_rate: -1 = automatic, 0 = no learning, >0 = manual rate
                if learning_rate != 0.0 {
                    let idx = self.sample_idx[row][col];
                    self.samples[row][col][idx % max_samples] = intensity;
                    self.sample_idx[row][col] = (idx + 1) % max_samples;
                }

                // Set foreground mask
                let fg_pixel = fgmask.at_mut(row, col)?;
                fg_pixel[0] = if is_background { 0 } else { 255 };
            }
        }

        Ok(())
    }

    /// Get background image
    pub fn get_background_image(&self, background: &mut Mat) -> Result<()> {
        if self.samples.is_empty() {
            return Err(Error::InvalidParameter(
                "Model not initialized".to_string(),
            ));
        }

        let rows = self.samples.len();
        let cols = self.samples[0].len();

        *background = Mat::new(rows, cols, 3, MatDepth::U8)?;

        for row in 0..rows {
            for col in 0..cols {
                // Median of samples
                let num_samples = self.sample_idx[row][col].min(self.samples[0][0].len());
                let mut samples: Vec<f32> = self.samples[row][col][0..num_samples].to_vec();

                if samples.is_empty() {
                    samples.push(0.0);
                }

                samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = samples[samples.len() / 2];

                let bg_pixel = background.at_mut(row, col)?;
                bg_pixel[0] = median as u8;
                bg_pixel[1] = median as u8;
                bg_pixel[2] = median as u8;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mog2_background_subtraction() {
        let mut subtractor = BackgroundSubtractorMOG2::new();

        let image = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut fgmask = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        subtractor.apply(&image, &mut fgmask, -1.0).unwrap();

        assert_eq!(fgmask.rows(), 100);
        assert_eq!(fgmask.cols(), 100);
        assert_eq!(fgmask.channels(), 1);
    }

    #[test]
    fn test_mog2_background_image() {
        let mut subtractor = BackgroundSubtractorMOG2::new();

        let image = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(100.0)).unwrap();
        let mut fgmask = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        // Process a few frames
        for _ in 0..10 {
            subtractor.apply(&image, &mut fgmask, -1.0).unwrap();
        }

        let mut background = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
        subtractor.get_background_image(&mut background).unwrap();

        assert_eq!(background.rows(), 50);
        assert_eq!(background.cols(), 50);
    }

    #[test]
    fn test_knn_background_subtraction() {
        let mut subtractor = BackgroundSubtractorKNN::new();

        let image = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut fgmask = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        subtractor.apply(&image, &mut fgmask, 0.1).unwrap();

        assert_eq!(fgmask.rows(), 100);
        assert_eq!(fgmask.cols(), 100);
        assert_eq!(fgmask.channels(), 1);
    }

    #[test]
    fn test_knn_background_image() {
        let mut subtractor = BackgroundSubtractorKNN::new();

        let image = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(100.0)).unwrap();
        let mut fgmask = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        // Process a few frames
        for _ in 0..10 {
            subtractor.apply(&image, &mut fgmask, 0.1).unwrap();
        }

        let mut background = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
        subtractor.get_background_image(&mut background).unwrap();

        assert_eq!(background.rows(), 50);
        assert_eq!(background.cols(), 50);
    }
}
