use crate::core::{Mat, MatDepth};
use crate::error::Result;

/// Seam finder for optimal image stitching seams
pub trait SeamFinder {
    fn find(&self, images: &[Mat], corners: &[(i32, i32)]) -> Result<Vec<Mat>>;
}

/// Graph Cut seam finder
pub struct GraphCutSeamFinder {
    cost_type: CostType,
    terminal_cost: f32,
    bad_region_penalty: f32,
}

#[derive(Clone, Copy)]
pub enum CostType {
    Color,      // Based on color difference
    ColorGrad,  // Color + gradient
}

impl Default for GraphCutSeamFinder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphCutSeamFinder {
    #[must_use] 
    pub fn new() -> Self {
        Self {
            cost_type: CostType::ColorGrad,
            terminal_cost: 10000.0,
            bad_region_penalty: 1000.0,
        }
    }

    #[must_use] 
    pub fn with_cost_type(mut self, cost_type: CostType) -> Self {
        self.cost_type = cost_type;
        self
    }
}

impl SeamFinder for GraphCutSeamFinder {
    fn find(&self, images: &[Mat], corners: &[(i32, i32)]) -> Result<Vec<Mat>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }

        let mut masks = Vec::new();

        for (i, img) in images.iter().enumerate() {
            let mask = Mat::new_with_default(
                img.rows(),
                img.cols(),
                1,
                MatDepth::U8,
                crate::core::types::Scalar::all(255.0),
            )?;
            masks.push(mask);
        }

        // Find seams between overlapping image pairs
        for i in 0..images.len() - 1 {
            let overlap = self.find_overlap(
                &images[i],
                &images[i + 1],
                corners[i],
                corners[i + 1],
            )?;

            if overlap.width > 0 {
                let seam = self.find_optimal_seam(&images[i], &images[i + 1], &overlap)?;
                // Use split_at_mut to get two mutable references safely
                let (left, right) = masks.split_at_mut(i + 1);
                self.apply_seam(&mut left[i], &mut right[0], &seam, &overlap)?;
            }
        }

        Ok(masks)
    }
}

impl GraphCutSeamFinder {
    fn find_overlap(
        &self,
        img1: &Mat,
        img2: &Mat,
        corner1: (i32, i32),
        corner2: (i32, i32),
    ) -> Result<OverlapRegion> {
        let x1_start = corner1.0;
        let x1_end = x1_start + img1.cols() as i32;
        let x2_start = corner2.0;
        let x2_end = x2_start + img2.cols() as i32;

        let overlap_x = x1_end.min(x2_end) - x1_start.max(x2_start);

        if overlap_x <= 0 {
            return Ok(OverlapRegion {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            });
        }

        let y1_start = corner1.1;
        let y1_end = y1_start + img1.rows() as i32;
        let y2_start = corner2.1;
        let y2_end = y2_start + img2.rows() as i32;

        let overlap_y = y1_end.min(y2_end) - y1_start.max(y2_start);

        Ok(OverlapRegion {
            x: x1_start.max(x2_start),
            y: y1_start.max(y2_start),
            width: overlap_x.max(0) as usize,
            height: overlap_y.max(0) as usize,
        })
    }

    fn find_optimal_seam(
        &self,
        img1: &Mat,
        img2: &Mat,
        overlap: &OverlapRegion,
    ) -> Result<Vec<usize>> {
        if overlap.height == 0 {
            return Ok(Vec::new());
        }

        // Build cost matrix
        let cost_mat = self.compute_cost_matrix(img1, img2, overlap)?;

        // Find minimum cost vertical seam using dynamic programming
        self.find_min_cost_seam(&cost_mat)
    }

    fn compute_cost_matrix(
        &self,
        img1: &Mat,
        img2: &Mat,
        overlap: &OverlapRegion,
    ) -> Result<Vec<Vec<f32>>> {
        let mut costs = vec![vec![0.0f32; overlap.width]; overlap.height];

        for row in 0..overlap.height {
            for col in 0..overlap.width {
                let cost = match self.cost_type {
                    CostType::Color => {
                        self.compute_color_cost(img1, img2, row, col, overlap)?
                    }
                    CostType::ColorGrad => {
                        let color_cost = self.compute_color_cost(img1, img2, row, col, overlap)?;
                        let grad_cost = self.compute_gradient_cost(img1, img2, row, col, overlap)?;
                        color_cost + grad_cost
                    }
                };

                costs[row][col] = cost;
            }
        }

        Ok(costs)
    }

    fn compute_color_cost(
        &self,
        img1: &Mat,
        img2: &Mat,
        row: usize,
        col: usize,
        _overlap: &OverlapRegion,
    ) -> Result<f32> {
        if row >= img1.rows() || col >= img1.cols() || row >= img2.rows() || col >= img2.cols() {
            return Ok(self.bad_region_penalty);
        }

        let mut cost = 0.0f32;

        for ch in 0..img1.channels().min(img2.channels()) {
            let val1 = f32::from(img1.at(row, col)?[ch]);
            let val2 = f32::from(img2.at(row, col)?[ch]);
            cost += (val1 - val2).abs();
        }

        Ok(cost)
    }

    fn compute_gradient_cost(
        &self,
        img1: &Mat,
        img2: &Mat,
        row: usize,
        col: usize,
        _overlap: &OverlapRegion,
    ) -> Result<f32> {
        if row == 0 || row >= img1.rows() - 1 || col == 0 || col >= img1.cols() - 1 {
            return Ok(0.0);
        }

        let mut grad_cost = 0.0f32;

        for ch in 0..img1.channels().min(img2.channels()) {
            // Gradient magnitude in img1
            let dx1 = f32::from(img1.at(row, col + 1)?[ch]) - f32::from(img1.at(row, col - 1)?[ch]);
            let dy1 = f32::from(img1.at(row + 1, col)?[ch]) - f32::from(img1.at(row - 1, col)?[ch]);
            let grad1 = (dx1 * dx1 + dy1 * dy1).sqrt();

            // Gradient magnitude in img2
            let dx2 = f32::from(img2.at(row, col + 1)?[ch]) - f32::from(img2.at(row, col - 1)?[ch]);
            let dy2 = f32::from(img2.at(row + 1, col)?[ch]) - f32::from(img2.at(row - 1, col)?[ch]);
            let grad2 = (dx2 * dx2 + dy2 * dy2).sqrt();

            grad_cost += (grad1 - grad2).abs();
        }

        Ok(grad_cost)
    }

    fn find_min_cost_seam(&self, costs: &[Vec<f32>]) -> Result<Vec<usize>> {
        let height = costs.len();
        if height == 0 {
            return Ok(Vec::new());
        }

        let width = costs[0].len();
        if width == 0 {
            return Ok(Vec::new());
        }

        // DP table
        let mut dp = vec![vec![f32::MAX; width]; height];
        let mut backtrack = vec![vec![0usize; width]; height];

        // Initialize first row
        for col in 0..width {
            dp[0][col] = costs[0][col];
        }

        // Fill DP table
        for row in 1..height {
            for col in 0..width {
                let mut min_cost = f32::MAX;
                let mut best_prev = col;

                for prev_col in col.saturating_sub(1)..=(col + 1).min(width - 1) {
                    if dp[row - 1][prev_col] < min_cost {
                        min_cost = dp[row - 1][prev_col];
                        best_prev = prev_col;
                    }
                }

                dp[row][col] = costs[row][col] + min_cost;
                backtrack[row][col] = best_prev;
            }
        }

        // Find minimum in last row
        let mut min_col = 0;
        let mut min_cost = dp[height - 1][0];

        for col in 1..width {
            if dp[height - 1][col] < min_cost {
                min_cost = dp[height - 1][col];
                min_col = col;
            }
        }

        // Backtrack
        let mut seam = vec![0usize; height];
        seam[height - 1] = min_col;

        for row in (0..height - 1).rev() {
            seam[row] = backtrack[row + 1][seam[row + 1]];
        }

        Ok(seam)
    }

    fn apply_seam(
        &self,
        mask1: &mut Mat,
        mask2: &mut Mat,
        seam: &[usize],
        overlap: &OverlapRegion,
    ) -> Result<()> {
        for (row_offset, &col) in seam.iter().enumerate() {
            let row = overlap.y as usize + row_offset;

            // Set mask1 to 0 on right side of seam
            for c in col..overlap.width {
                if row < mask1.rows() && c < mask1.cols() {
                    mask1.at_mut(row, c)?[0] = 0;
                }
            }

            // Set mask2 to 0 on left side of seam
            for c in 0..col {
                if row < mask2.rows() && c < mask2.cols() {
                    mask2.at_mut(row, c)?[0] = 0;
                }
            }
        }

        Ok(())
    }
}

/// Voronoi seam finder (simpler, faster alternative)
pub struct VoronoiSeamFinder;

impl Default for VoronoiSeamFinder {
    fn default() -> Self {
        Self::new()
    }
}

impl VoronoiSeamFinder {
    #[must_use] 
    pub fn new() -> Self {
        Self
    }
}

impl SeamFinder for VoronoiSeamFinder {
    fn find(&self, images: &[Mat], corners: &[(i32, i32)]) -> Result<Vec<Mat>> {
        let mut masks = Vec::new();

        for img in images {
            let mask = Mat::new_with_default(
                img.rows(),
                img.cols(),
                1,
                MatDepth::U8,
                crate::core::types::Scalar::all(255.0),
            )?;
            masks.push(mask);
        }

        // Simple Voronoi partitioning based on image centers
        if images.len() >= 2 {
            for i in 0..masks.len() {
                let center_x = corners[i].0 + images[i].cols() as i32 / 2;

                for row in 0..masks[i].rows() {
                    for col in 0..masks[i].cols() {
                        let pixel_x = corners[i].0 + col as i32;

                        // Check if closer to another image center
                        for j in 0..images.len() {
                            if i != j {
                                let other_center_x = corners[j].0 + images[j].cols() as i32 / 2;
                                let dist_to_current = (pixel_x - center_x).abs();
                                let dist_to_other = (pixel_x - other_center_x).abs();

                                if dist_to_other < dist_to_current {
                                    masks[i].at_mut(row, col)?[0] = 0;
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(masks)
    }
}

struct OverlapRegion {
    x: i32,
    y: i32,
    width: usize,
    height: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_graph_cut_seam_finder() {
        let img1 = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(100.0)).unwrap();
        let img2 = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(150.0)).unwrap();

        let images = vec![img1, img2];
        let corners = vec![(0, 0), (50, 0)];

        let finder = GraphCutSeamFinder::new();
        let masks = finder.find(&images, &corners).unwrap();

        assert_eq!(masks.len(), 2);
        assert_eq!(masks[0].channels(), 1);
    }

    #[test]
    fn test_voronoi_seam_finder() {
        let img1 = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(100.0)).unwrap();
        let img2 = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(150.0)).unwrap();

        let images = vec![img1, img2];
        let corners = vec![(0, 0), (50, 0)];

        let finder = VoronoiSeamFinder::new();
        let masks = finder.find(&images, &corners).unwrap();

        assert_eq!(masks.len(), 2);
    }
}
