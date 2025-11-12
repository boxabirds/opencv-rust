use crate::core::{Mat, MatDepth};
use crate::core::types::Rect;
use crate::error::{Error, Result};

/// Background subtractor using MOG2 algorithm (simplified)
pub struct BackgroundSubtractorMOG2 {
    history: Vec<Mat>,
    max_history: usize,
    var_threshold: f64,
}

impl BackgroundSubtractorMOG2 {
    #[must_use]
    pub fn new(history: i32, var_threshold: f64) -> Self {
        #[allow(clippy::cast_sign_loss)]
        let max_history = history as usize;
        Self {
            history: Vec::new(),
            max_history,
            var_threshold,
        }
    }

    pub fn apply(&mut self, image: &Mat, learning_rate: f64) -> Result<Mat> {
        if image.depth() != MatDepth::U8 {
            return Err(Error::UnsupportedOperation(
                "MOG2 only supports U8 depth".to_string(),
            ));
        }

        // Add current frame to history
        self.history.push(image.clone_mat());

        if self.history.len() > self.max_history {
            self.history.remove(0);
        }

        // Create foreground mask
        let mut fg_mask = Mat::new(image.rows(), image.cols(), 1, MatDepth::U8)?;

        if self.history.len() < 2 {
            // Not enough history, return empty mask
            return Ok(fg_mask);
        }

        // Simple background subtraction: compare with median of history
        for row in 0..image.rows() {
            for col in 0..image.cols() {
                let current_pixel = image.at(row, col)?;

                // Collect history values for this pixel
                let mut history_vals = Vec::new();

                for hist_img in &self.history {
                    if let Ok(hist_pixel) = hist_img.at(row, col) {
                        let val = if image.channels() == 1 {
                            f64::from(hist_pixel[0])
                        } else {
                            // Use brightness for multi-channel
                            (f64::from(hist_pixel[0]) + f64::from(hist_pixel[1]) + f64::from(hist_pixel[2])) / 3.0
                        };
                        history_vals.push(val);
                    }
                }

                // Calculate median
                history_vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = if history_vals.is_empty() {
                    0.0
                } else {
                    history_vals[history_vals.len() / 2]
                };

                // Compare current with median
                let current_val = if image.channels() == 1 {
                    f64::from(current_pixel[0])
                } else {
                    (f64::from(current_pixel[0]) + f64::from(current_pixel[1]) + f64::from(current_pixel[2])) / 3.0
                };

                let diff = (current_val - median).abs();

                let fg_pixel = fg_mask.at_mut(row, col)?;
                fg_pixel[0] = if diff > self.var_threshold { 255 } else { 0 };
            }
        }

        Ok(fg_mask)
    }
}

/// `MeanShift` tracker
pub struct MeanShiftTracker {
    window: Rect,
    max_iterations: i32,
    epsilon: f64,
}

impl MeanShiftTracker {
    #[must_use] 
    pub fn new(window: Rect) -> Self {
        Self {
            window,
            max_iterations: 100,
            epsilon: 1.0,
        }
    }

    pub fn track(&mut self, prob_image: &Mat) -> Result<Rect> {
        if prob_image.channels() != 1 {
            return Err(Error::InvalidParameter(
                "MeanShift requires single-channel probability image".to_string(),
            ));
        }

        let mut current_window = self.window;

        for _ in 0..self.max_iterations {
            // Calculate centroid of probability distribution in window
            let mut sum_x = 0.0;
            let mut sum_y = 0.0;
            let mut sum_weight = 0.0;

            for y in current_window.y..(current_window.y + current_window.height) {
                for x in current_window.x..(current_window.x + current_window.width) {
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let rows_i32 = prob_image.rows() as i32;
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let cols_i32 = prob_image.cols() as i32;

                    if y >= 0 && y < rows_i32 && x >= 0 && x < cols_i32 {
                        #[allow(clippy::cast_sign_loss)]
                        let y_usize = y as usize;
                        #[allow(clippy::cast_sign_loss)]
                        let x_usize = x as usize;
                        let pixel = prob_image.at(y_usize, x_usize)?;
                        let weight = f64::from(pixel[0]);

                        sum_x += f64::from(x) * weight;
                        sum_y += f64::from(y) * weight;
                        sum_weight += weight;
                    }
                }
            }

            if sum_weight == 0.0 {
                break;
            }

            #[allow(clippy::cast_possible_truncation)]
            let centroid_x = (sum_x / sum_weight) as i32;
            #[allow(clippy::cast_possible_truncation)]
            let centroid_y = (sum_y / sum_weight) as i32;

            // Move window to centroid
            let new_x = centroid_x - current_window.width / 2;
            let new_y = centroid_y - current_window.height / 2;

            let shift = f64::from((new_x - current_window.x).pow(2) + (new_y - current_window.y).pow(2));

            current_window.x = new_x;
            current_window.y = new_y;

            if shift.sqrt() < self.epsilon {
                break;
            }
        }

        self.window = current_window;
        Ok(current_window)
    }
}

/// `CamShift` tracker (continuously adaptive mean shift)
pub struct CamShiftTracker {
    mean_shift: MeanShiftTracker,
}

impl CamShiftTracker {
    #[must_use] 
    pub fn new(window: Rect) -> Self {
        Self {
            mean_shift: MeanShiftTracker::new(window),
        }
    }

    pub fn track(&mut self, prob_image: &Mat) -> Result<Rect> {
        // First apply mean shift
        let new_window = self.mean_shift.track(prob_image)?;

        // Calculate moments to adjust window size (simplified)
        // In full CamShift, we would calculate orientation and size from moments

        Ok(new_window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_background_subtractor() {
        let mut subtractor = BackgroundSubtractorMOG2::new(10, 16.0);

        let frame1 = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let frame2 = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(130.0)).unwrap();

        let _ = subtractor.apply(&frame1, -1.0).unwrap();
        let fg_mask = subtractor.apply(&frame2, -1.0).unwrap();

        assert_eq!(fg_mask.rows(), frame1.rows());
    }

    #[test]
    fn test_meanshift_tracker() {
        let prob_image = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut tracker = MeanShiftTracker::new(Rect::new(40, 40, 20, 20));

        let result = tracker.track(&prob_image).unwrap();
        assert!(result.width > 0 && result.height > 0);
    }
}
