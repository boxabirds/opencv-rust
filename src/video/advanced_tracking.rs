use crate::core::{Mat, MatDepth};
use crate::core::types::{Rect, Point};
use crate::error::{Error, Result};

/// MOSSE (Minimum Output Sum of Squared Error) tracker
pub struct MOSSETracker {
    template: Mat,
    filter: Vec<Vec<f32>>,
    learning_rate: f32,
    initialized: bool,
}

impl Default for MOSSETracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MOSSETracker {
    /// Create MOSSE tracker with default parameters
    #[must_use] 
    pub fn new() -> Self {
        Self {
            template: Mat::new(1, 1, 1, MatDepth::U8).unwrap(),
            filter: Vec::new(),
            learning_rate: 0.125,
            initialized: false,
        }
    }

    /// Initialize tracker with first frame and bounding box
    pub fn init(&mut self, frame: &Mat, bbox: Rect) -> Result<()> {
        if frame.channels() != 1 {
            return Err(Error::InvalidParameter("MOSSE requires grayscale image".to_string()));
        }

        // Extract template from bounding box
        self.template = extract_patch(frame, bbox)?;

        // Initialize correlation filter
        let h = self.template.rows();
        let w = self.template.cols();
        self.filter = vec![vec![1.0; w]; h];

        // Create Gaussian response
        let gaussian = create_gaussian_response(h, w, 2.0);

        // Initialize filter
        let template_clone = self.template.clone();
        self.update_filter(&template_clone, &gaussian)?;

        self.initialized = true;
        Ok(())
    }

    /// Update tracker with new frame
    pub fn update(&mut self, frame: &Mat) -> Result<Rect> {
        if !self.initialized {
            return Err(Error::InvalidParameter("Tracker not initialized".to_string()));
        }

        if frame.channels() != 1 {
            return Err(Error::InvalidParameter("MOSSE requires grayscale image".to_string()));
        }

        // Search for best match using correlation filter
        let response = self.apply_filter(frame)?;

        // Find peak in response
        let (max_row, max_col, _) = find_max(&response)?;

        // Update filter with new location
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        let max_col_i32 = max_col as i32;
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        let max_row_i32 = max_row as i32;
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        let template_cols_i32 = self.template.cols() as i32;
        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
        let template_rows_i32 = self.template.rows() as i32;

        let new_bbox = Rect::new(
            max_col_i32 - template_cols_i32 / 2,
            max_row_i32 - template_rows_i32 / 2,
            template_cols_i32,
            template_rows_i32,
        );

        let new_template = extract_patch(frame, new_bbox)?;
        let gaussian = create_gaussian_response(self.template.rows(), self.template.cols(), 2.0);
        self.update_filter(&new_template, &gaussian)?;

        Ok(new_bbox)
    }

    fn apply_filter(&self, frame: &Mat) -> Result<Mat> {
        let mut response = Mat::new(frame.rows(), frame.cols(), 1, MatDepth::U8)?;

        let fh = self.filter.len();
        let fw = self.filter[0].len();

        for row in 0..frame.rows() {
            for col in 0..frame.cols() {
                let mut sum = 0.0f32;

                for fy in 0..fh {
                    for fx in 0..fw {
                        let y = (row + fy).min(frame.rows() - 1);
                        let x = (col + fx).min(frame.cols() - 1);

                        let pixel = f32::from(frame.at(y, x)?[0]);
                        sum += pixel * self.filter[fy][fx];
                    }
                }

                let resp_pixel = response.at_mut(row, col)?;
                let clamped = sum.abs().min(255.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let pixel_val = clamped as u8;
                resp_pixel[0] = pixel_val;
            }
        }

        Ok(response)
    }

    fn update_filter(&mut self, template: &Mat, gaussian: &[Vec<f32>]) -> Result<()> {
        let h = template.rows();
        let w = template.cols();

        for row in 0..h {
            for col in 0..w {
                let pixel = f32::from(template.at(row, col)?[0]) / 255.0;
                let g = gaussian[row][col];

                // Update filter using learning rate
                self.filter[row][col] =
                    (1.0 - self.learning_rate) * self.filter[row][col] + self.learning_rate * g * pixel;
            }
        }

        Ok(())
    }
}

/// `MedianFlow` tracker
pub struct MedianFlowTracker {
    points: Vec<Point>,
    bbox: Rect,
    initialized: bool,
}

impl Default for MedianFlowTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl MedianFlowTracker {
    /// Create `MedianFlow` tracker
    #[must_use] 
    pub fn new() -> Self {
        Self {
            points: Vec::new(),
            bbox: Rect::new(0, 0, 0, 0),
            initialized: false,
        }
    }

    /// Initialize tracker
    pub fn init(&mut self, _frame: &Mat, bbox: Rect) -> Result<()> {
        self.bbox = bbox;

        // Generate grid of points in bounding box
        self.points.clear();
        let grid_size = 10;

        for i in 0..grid_size {
            for j in 0..grid_size {
                let x = bbox.x + (bbox.width * i) / grid_size;
                let y = bbox.y + (bbox.height * j) / grid_size;
                self.points.push(Point::new(x, y));
            }
        }

        self.initialized = true;
        Ok(())
    }

    /// Update tracker
    pub fn update(&mut self, _prev_frame: &Mat, _curr_frame: &Mat) -> Result<Rect> {
        if !self.initialized {
            return Err(Error::InvalidParameter("Tracker not initialized".to_string()));
        }

        // Track points using optical flow (simplified)
        // In real implementation, would use Lucas-Kanade

        // Compute median displacement
        let mut displacements_x = Vec::new();
        let mut displacements_y = Vec::new();

        // Simplified: assume small motion
        for _point in &self.points {
            displacements_x.push(0);
            displacements_y.push(0);
        }

        displacements_x.sort_unstable();
        displacements_y.sort_unstable();

        let median_dx = displacements_x[displacements_x.len() / 2];
        let median_dy = displacements_y[displacements_y.len() / 2];

        // Update bounding box
        self.bbox.x += median_dx;
        self.bbox.y += median_dy;

        // Update points
        for point in &mut self.points {
            point.x += median_dx;
            point.y += median_dy;
        }

        Ok(self.bbox)
    }
}

/// CSRT (Discriminative Correlation Filter with Channel and Spatial Reliability) tracker
pub struct CSRTTracker {
    template: Mat,
    bbox: Rect,
    learning_rate: f32,
    padding: f32,
    initialized: bool,
}

impl Default for CSRTTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CSRTTracker {
    /// Create CSRT tracker
    #[must_use] 
    pub fn new() -> Self {
        Self {
            template: Mat::new(1, 1, 1, MatDepth::U8).unwrap(),
            bbox: Rect::new(0, 0, 0, 0),
            learning_rate: 0.02,
            padding: 2.0,
            initialized: false,
        }
    }

    /// Initialize tracker
    pub fn init(&mut self, frame: &Mat, bbox: Rect) -> Result<()> {
        self.bbox = bbox;

        // Extract padded template
        #[allow(clippy::cast_precision_loss)]
        let bbox_x_f32 = bbox.x as f32;
        #[allow(clippy::cast_precision_loss)]
        let bbox_y_f32 = bbox.y as f32;
        #[allow(clippy::cast_precision_loss)]
        let bbox_width_f32 = bbox.width as f32;
        #[allow(clippy::cast_precision_loss)]
        let bbox_height_f32 = bbox.height as f32;

        #[allow(clippy::cast_possible_truncation)]
        let padded_x = (bbox_x_f32 - bbox_width_f32 * (self.padding - 1.0) / 2.0) as i32;
        #[allow(clippy::cast_possible_truncation)]
        let padded_y = (bbox_y_f32 - bbox_height_f32 * (self.padding - 1.0) / 2.0) as i32;
        #[allow(clippy::cast_possible_truncation)]
        let padded_width = (bbox_width_f32 * self.padding) as i32;
        #[allow(clippy::cast_possible_truncation)]
        let padded_height = (bbox_height_f32 * self.padding) as i32;

        let padded_bbox = Rect::new(padded_x, padded_y, padded_width, padded_height);

        self.template = extract_patch(frame, padded_bbox)?;
        self.initialized = true;
        Ok(())
    }

    /// Update tracker
    pub fn update(&mut self, frame: &Mat) -> Result<Rect> {
        if !self.initialized {
            return Err(Error::InvalidParameter("Tracker not initialized".to_string()));
        }

        // Search in region around previous location
        let search_region = Rect::new(
            self.bbox.x - self.bbox.width / 2,
            self.bbox.y - self.bbox.height / 2,
            self.bbox.width * 2,
            self.bbox.height * 2,
        );

        // Find best match using template matching
        let (best_x, best_y, _) = template_match(frame, &self.template, search_region)?;

        // Update bounding box
        self.bbox.x = best_x;
        self.bbox.y = best_y;

        // Update template with learning rate
        let new_template = extract_patch(frame, self.bbox)?;
        self.template = blend_templates(&self.template, &new_template, self.learning_rate)?;

        Ok(self.bbox)
    }
}

// Helper functions

fn extract_patch(frame: &Mat, bbox: Rect) -> Result<Mat> {
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let frame_cols_i32 = frame.cols() as i32;
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let frame_rows_i32 = frame.rows() as i32;

    #[allow(clippy::cast_sign_loss)]
    let x = bbox.x.max(0).min(frame_cols_i32 - 1) as usize;
    #[allow(clippy::cast_sign_loss)]
    let y = bbox.y.max(0).min(frame_rows_i32 - 1) as usize;

    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let x_i32 = x as i32;
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let y_i32 = y as i32;

    #[allow(clippy::cast_sign_loss)]
    let w = bbox.width.max(1).min(frame_cols_i32 - x_i32) as usize;
    #[allow(clippy::cast_sign_loss)]
    let h = bbox.height.max(1).min(frame_rows_i32 - y_i32) as usize;

    let mut patch = Mat::new(h, w, frame.channels(), frame.depth())?;

    for row in 0..h {
        for col in 0..w {
            let src_pixel = frame.at(y + row, x + col)?;
            let dst_pixel = patch.at_mut(row, col)?;
            for ch in 0..frame.channels() {
                dst_pixel[ch] = src_pixel[ch];
            }
        }
    }

    Ok(patch)
}

fn create_gaussian_response(height: usize, width: usize, sigma: f32) -> Vec<Vec<f32>> {
    let mut gaussian = vec![vec![0.0; width]; height];

    #[allow(clippy::cast_precision_loss)]
    let center_y = height as f32 / 2.0;
    #[allow(clippy::cast_precision_loss)]
    let center_x = width as f32 / 2.0;

    for y in 0..height {
        for x in 0..width {
            #[allow(clippy::cast_precision_loss)]
            let dy = y as f32 - center_y;
            #[allow(clippy::cast_precision_loss)]
            let dx = x as f32 - center_x;
            let dist2 = dx * dx + dy * dy;

            gaussian[y][x] = (-dist2 / (2.0 * sigma * sigma)).exp();
        }
    }

    gaussian
}

fn find_max(mat: &Mat) -> Result<(usize, usize, u8)> {
    let mut max_val = 0u8;
    let mut max_row = 0;
    let mut max_col = 0;

    for row in 0..mat.rows() {
        for col in 0..mat.cols() {
            let pixel = mat.at(row, col)?;
            if pixel[0] > max_val {
                max_val = pixel[0];
                max_row = row;
                max_col = col;
            }
        }
    }

    Ok((max_row, max_col, max_val))
}

fn template_match(frame: &Mat, template: &Mat, search_region: Rect) -> Result<(i32, i32, f32)> {
    #[allow(clippy::cast_sign_loss)]
    let x_start = search_region.x.max(0) as usize;
    #[allow(clippy::cast_sign_loss)]
    let y_start = search_region.y.max(0) as usize;

    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let frame_cols_i32 = frame.cols() as i32;
    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
    let frame_rows_i32 = frame.rows() as i32;

    #[allow(clippy::cast_sign_loss)]
    let x_end = (search_region.x + search_region.width).min(frame_cols_i32) as usize;
    #[allow(clippy::cast_sign_loss)]
    let y_end = (search_region.y + search_region.height).min(frame_rows_i32) as usize;

    let mut best_score = f32::MAX;
    let mut best_x = search_region.x;
    let mut best_y = search_region.y;

    for y in y_start..y_end.saturating_sub(template.rows()) {
        for x in x_start..x_end.saturating_sub(template.cols()) {
            // Compute SSD (sum of squared differences)
            let mut ssd = 0.0f32;

            for ty in 0..template.rows() {
                for tx in 0..template.cols() {
                    let frame_pixel = frame.at(y + ty, x + tx)?;
                    let template_pixel = template.at(ty, tx)?;

                    for ch in 0..frame.channels().min(template.channels()) {
                        let diff = f32::from(frame_pixel[ch]) - f32::from(template_pixel[ch]);
                        ssd += diff * diff;
                    }
                }
            }

            if ssd < best_score {
                best_score = ssd;
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                let x_i32 = x as i32;
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                let y_i32 = y as i32;
                best_x = x_i32;
                best_y = y_i32;
            }
        }
    }

    Ok((best_x, best_y, best_score))
}

fn blend_templates(old_template: &Mat, new_template: &Mat, alpha: f32) -> Result<Mat> {
    if old_template.rows() != new_template.rows() || old_template.cols() != new_template.cols() {
        return Ok(new_template.clone());
    }

    let mut blended = Mat::new(
        old_template.rows(),
        old_template.cols(),
        old_template.channels(),
        old_template.depth(),
    )?;

    for row in 0..old_template.rows() {
        for col in 0..old_template.cols() {
            let old_pixel = old_template.at(row, col)?;
            let new_pixel = new_template.at(row, col)?;
            let blended_pixel = blended.at_mut(row, col)?;

            for ch in 0..old_template.channels() {
                let value = (1.0 - alpha) * f32::from(old_pixel[ch]) + alpha * f32::from(new_pixel[ch]);
                let clamped = value.clamp(0.0, 255.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let pixel_val = clamped as u8;
                blended_pixel[ch] = pixel_val;
            }
        }
    }

    Ok(blended)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_mosse_tracker() {
        let frame = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let bbox = Rect::new(40, 40, 20, 20);

        let mut tracker = MOSSETracker::new();
        tracker.init(&frame, bbox).unwrap();

        let new_bbox = tracker.update(&frame).unwrap();
        assert!(new_bbox.width > 0);
        assert!(new_bbox.height > 0);
    }

    #[test]
    fn test_medianflow_tracker() {
        let frame = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let bbox = Rect::new(40, 40, 20, 20);

        let mut tracker = MedianFlowTracker::new();
        tracker.init(&frame, bbox).unwrap();

        let new_bbox = tracker.update(&frame, &frame).unwrap();
        assert_eq!(new_bbox.width, 20);
        assert_eq!(new_bbox.height, 20);
    }

    #[test]
    fn test_csrt_tracker() {
        let frame = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let bbox = Rect::new(40, 40, 20, 20);

        let mut tracker = CSRTTracker::new();
        tracker.init(&frame, bbox).unwrap();

        let new_bbox = tracker.update(&frame).unwrap();
        assert!(new_bbox.width > 0);
        assert!(new_bbox.height > 0);
    }

    #[test]
    fn test_extract_patch() {
        let frame = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let bbox = Rect::new(10, 10, 20, 20);

        let patch = extract_patch(&frame, bbox).unwrap();
        assert_eq!(patch.rows(), 20);
        assert_eq!(patch.cols(), 20);
    }

    #[test]
    fn test_create_gaussian_response() {
        let gaussian = create_gaussian_response(10, 10, 2.0);
        assert_eq!(gaussian.len(), 10);
        assert_eq!(gaussian[0].len(), 10);

        // Center should have highest value
        assert!(gaussian[5][5] > gaussian[0][0]);
    }
}
