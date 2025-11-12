use crate::core::{Mat, MatDepth};
use crate::error::Result;

/// Seam carving for content-aware image resizing
pub struct SeamCarver {
    energy_type: EnergyType,
}

#[derive(Clone, Copy)]
pub enum EnergyType {
    Gradient,      // Gradient magnitude
    Laplacian,     // Laplacian of Gaussian
    ForwardEnergy, // Forward energy (considers new edges)
}

impl SeamCarver {
    #[must_use] 
    pub fn new(energy_type: EnergyType) -> Self {
        Self { energy_type }
    }

    /// Reduce width by removing vertical seams
    pub fn reduce_width(&self, src: &Mat, target_width: usize) -> Result<Mat> {
        if target_width >= src.cols() {
            return Ok(src.clone_mat());
        }

        let seams_to_remove = src.cols() - target_width;
        let mut current = src.clone_mat();

        for _ in 0..seams_to_remove {
            current = self.remove_vertical_seam(&current)?;
        }

        Ok(current)
    }

    /// Reduce height by removing horizontal seams
    pub fn reduce_height(&self, src: &Mat, target_height: usize) -> Result<Mat> {
        if target_height >= src.rows() {
            return Ok(src.clone_mat());
        }

        let seams_to_remove = src.rows() - target_height;
        let mut current = src.clone_mat();

        for _ in 0..seams_to_remove {
            current = self.remove_horizontal_seam(&current)?;
        }

        Ok(current)
    }

    /// Resize to target dimensions using seam carving
    pub fn resize(&self, src: &Mat, target_width: usize, target_height: usize) -> Result<Mat> {
        let mut result = self.reduce_width(src, target_width)?;
        result = self.reduce_height(&result, target_height)?;
        Ok(result)
    }

    fn remove_vertical_seam(&self, src: &Mat) -> Result<Mat> {
        // Compute energy map
        let energy = self.compute_energy(src)?;

        // Find seam path using dynamic programming
        let seam = self.find_vertical_seam(&energy)?;

        // Remove seam
        let new_width = src.cols() - 1;
        let mut result = Mat::new(src.rows(), new_width, src.channels(), src.depth())?;

        for row in 0..src.rows() {
            let seam_col = seam[row];
            let mut dst_col = 0;

            for src_col in 0..src.cols() {
                if src_col != seam_col {
                    #[allow(clippy::cast_possible_truncation)]
                    for ch in 0..src.channels() {
                        result.at_mut(row, dst_col)?[ch] = src.at(row, src_col)?[ch];
                    }
                    dst_col += 1;
                }
            }
        }

        Ok(result)
    }

    fn remove_horizontal_seam(&self, src: &Mat) -> Result<Mat> {
        // Transpose, remove vertical seam, transpose back
        let transposed = self.transpose(src)?;
        let removed = self.remove_vertical_seam(&transposed)?;
        self.transpose(&removed)
    }

    fn transpose(&self, src: &Mat) -> Result<Mat> {
        let mut result = Mat::new(src.cols(), src.rows(), src.channels(), src.depth())?;

        for row in 0..src.rows() {
            for col in 0..src.cols() {
                #[allow(clippy::cast_possible_truncation)]
                for ch in 0..src.channels() {
                    result.at_mut(col, row)?[ch] = src.at(row, col)?[ch];
                }
            }
        }

        Ok(result)
    }

    fn compute_energy(&self, src: &Mat) -> Result<Mat> {
        let mut energy = Mat::new(src.rows(), src.cols(), 1, MatDepth::F32)?;

        match self.energy_type {
            EnergyType::Gradient => self.compute_gradient_energy(src, &mut energy)?,
            EnergyType::Laplacian => self.compute_laplacian_energy(src, &mut energy)?,
            EnergyType::ForwardEnergy => self.compute_gradient_energy(src, &mut energy)?,
        }

        Ok(energy)
    }

    fn compute_gradient_energy(&self, src: &Mat, energy: &mut Mat) -> Result<()> {
        for row in 0..src.rows() {
            for col in 0..src.cols() {
                let mut dx = 0.0f32;
                let mut dy = 0.0f32;

                // Horizontal gradient
                if col > 0 && col < src.cols() - 1 {
                    #[allow(clippy::cast_possible_truncation)]
                    for ch in 0..src.channels() {
                        let left = f32::from(src.at(row, col - 1)?[ch]);
                        let right = f32::from(src.at(row, col + 1)?[ch]);
                        dx += (right - left).abs();
                    }
                }

                // Vertical gradient
                if row > 0 && row < src.rows() - 1 {
                    #[allow(clippy::cast_possible_truncation)]
                    for ch in 0..src.channels() {
                        let up = f32::from(src.at(row - 1, col)?[ch]);
                        let down = f32::from(src.at(row + 1, col)?[ch]);
                        dy += (down - up).abs();
                    }
                }

                energy.set_f32(row, col, 0, dx + dy)?;
            }
        }

        Ok(())
    }

    fn compute_laplacian_energy(&self, src: &Mat, energy: &mut Mat) -> Result<()> {
        for row in 1..src.rows() - 1 {
            for col in 1..src.cols() - 1 {
                let mut laplacian = 0.0f32;

                #[allow(clippy::cast_possible_truncation)]
                for ch in 0..src.channels() {
                    let center = f32::from(src.at(row, col)?[ch]);
                    let left = f32::from(src.at(row, col - 1)?[ch]);
                    let right = f32::from(src.at(row, col + 1)?[ch]);
                    let up = f32::from(src.at(row - 1, col)?[ch]);
                    let down = f32::from(src.at(row + 1, col)?[ch]);

                    laplacian += (left + right + up + down - 4.0 * center).abs();
                }

                energy.set_f32(row, col, 0, laplacian)?;
            }
        }

        Ok(())
    }

    fn find_vertical_seam(&self, energy: &Mat) -> Result<Vec<usize>> {
        let rows = energy.rows();
        let cols = energy.cols();

        // DP table
        let mut dp = Mat::new(rows, cols, 1, MatDepth::F32)?;
        let mut backtrack = vec![vec![0usize; cols]; rows];

        // Initialize first row
        for col in 0..cols {
            dp.set_f32(0, col, 0, energy.at_f32(0, col, 0)?)?;
        }

        // Fill DP table
        for row in 1..rows {
            for col in 0..cols {
                let mut min_energy = f32::MAX;
                let mut best_prev_col = col;

                // Check three possible previous positions
                for prev_col in col.saturating_sub(1)..=(col + 1).min(cols - 1) {
                    let prev_energy = dp.at_f32(row - 1, prev_col, 0)?;
                    if prev_energy < min_energy {
                        min_energy = prev_energy;
                        best_prev_col = prev_col;
                    }
                }

                let current_energy = energy.at_f32(row, col, 0)?;
                dp.set_f32(row, col, 0, min_energy + current_energy)?;
                backtrack[row][col] = best_prev_col;
            }
        }

        // Find minimum in last row
        let mut min_col = 0;
        let mut min_energy = dp.at_f32(rows - 1, 0, 0)?;

        for col in 1..cols {
            let energy_val = dp.at_f32(rows - 1, col, 0)?;
            if energy_val < min_energy {
                min_energy = energy_val;
                min_col = col;
            }
        }

        // Backtrack to find seam path
        let mut seam = vec![0usize; rows];
        seam[rows - 1] = min_col;

        for row in (0..rows - 1).rev() {
            seam[row] = backtrack[row + 1][seam[row + 1]];
        }

        Ok(seam)
    }
}

/// Enlarge image using seam insertion
pub fn enlarge_width(src: &Mat, target_width: usize) -> Result<Mat> {
    if target_width <= src.cols() {
        return Ok(src.clone_mat());
    }

    let carver = SeamCarver::new(EnergyType::Gradient);
    let seams_to_add = target_width - src.cols();

    // Find seams to duplicate
    let energy = carver.compute_energy(src)?;
    let mut seams = Vec::new();

    for _ in 0..seams_to_add.min(10) {
        // Limit number of seams
        let seam = carver.find_vertical_seam(&energy)?;
        seams.push(seam);
    }

    // Duplicate seams
    let mut result = src.clone_mat();
    for seam in seams {
        result = duplicate_vertical_seam(&result, &seam)?;
    }

    Ok(result)
}

fn duplicate_vertical_seam(src: &Mat, seam: &[usize]) -> Result<Mat> {
    let new_width = src.cols() + 1;
    let mut result = Mat::new(src.rows(), new_width, src.channels(), src.depth())?;

    for row in 0..src.rows() {
        let seam_col = seam[row];
        let mut dst_col = 0;

        for src_col in 0..src.cols() {
            // Copy original pixel
            #[allow(clippy::cast_possible_truncation)]
            for ch in 0..src.channels() {
                result.at_mut(row, dst_col)?[ch] = src.at(row, src_col)?[ch];
            }
            dst_col += 1;

            // Duplicate seam pixel
            if src_col == seam_col {
                #[allow(clippy::cast_possible_truncation)]
                for ch in 0..src.channels() {
                    result.at_mut(row, dst_col)?[ch] = src.at(row, src_col)?[ch];
                }
                dst_col += 1;
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_seam_carving_width() {
        let src = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let carver = SeamCarver::new(EnergyType::Gradient);
        let result = carver.reduce_width(&src, 90).unwrap();

        assert_eq!(result.cols(), 90);
        assert_eq!(result.rows(), 100);
    }

    #[test]
    fn test_seam_carving_height() {
        let src = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let carver = SeamCarver::new(EnergyType::Gradient);
        let result = carver.reduce_height(&src, 90).unwrap();

        assert_eq!(result.rows(), 90);
        assert_eq!(result.cols(), 100);
    }

    #[test]
    fn test_seam_carving_resize() {
        let src = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let carver = SeamCarver::new(EnergyType::Laplacian);
        let result = carver.resize(&src, 80, 90).unwrap();

        assert_eq!(result.cols(), 80);
        assert_eq!(result.rows(), 90);
    }

    #[test]
    fn test_enlarge_width() {
        let src = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let result = enlarge_width(&src, 60).unwrap();

        assert_eq!(result.cols(), 60);
        assert_eq!(result.rows(), 50);
    }
}
