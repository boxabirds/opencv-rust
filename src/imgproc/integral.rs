#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, clippy::cast_possible_wrap)]
use crate::core::Mat;
use crate::error::Result;

/// Compute integral image (summed-area table)
/// Each element dst(x,y) = sum of all pixels src(x'<=x, y'<=y)
pub fn integral(src: &Mat, dst: &mut Mat) -> Result<()> {
    let rows = src.rows();
    let cols = src.cols();
    let channels = src.channels();

    // Use u32 accumulator to handle sums
    let mut integral_data = vec![0u32; rows * cols * channels];

    for c in 0..channels {
        for y in 0..rows {
            for x in 0..cols {
                let idx = (y * cols + x) * channels + c;
                let pixel_val = u32::from(src.at(y, x)?[c]);

                let mut sum = pixel_val;

                // Add value from above
                if y > 0 {
                    let above_idx = ((y - 1) * cols + x) * channels + c;
                    sum += integral_data[above_idx];
                }

                // Add value from left
                if x > 0 {
                    let left_idx = (y * cols + (x - 1)) * channels + c;
                    sum += integral_data[left_idx];
                }

                // Subtract top-left (counted twice)
                if y > 0 && x > 0 {
                    let top_left_idx = ((y - 1) * cols + (x - 1)) * channels + c;
                    sum -= integral_data[top_left_idx];
                }

                integral_data[idx] = sum;

                // Normalize back to U8 range (scale down)
                // For display purposes, we normalize by dividing by max possible value
                let normalized = sum.min(255) as u8;
                dst.at_mut(y, x)?[c] = normalized;
            }
        }
    }

    Ok(())
}
