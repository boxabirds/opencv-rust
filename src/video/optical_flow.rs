use crate::core::{Mat, MatDepth};
use crate::core::types::{Point, Size};
use crate::error::{Error, Result};

/// Calculate optical flow using Lucas-Kanade method
pub fn calc_optical_flow_pyr_lk(
    prev_img: &Mat,
    next_img: &Mat,
    prev_pts: &[Point],
    win_size: Size,
    max_level: i32,
) -> Result<(Vec<Point>, Vec<u8>)> {
    if prev_img.channels() != 1 || next_img.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Optical flow requires grayscale images".to_string(),
        ));
    }

    if prev_img.rows() != next_img.rows() || prev_img.cols() != next_img.cols() {
        return Err(Error::InvalidDimensions(
            "Images must have same dimensions".to_string(),
        ));
    }

    let mut next_pts = Vec::new();
    let mut status = Vec::new();

    let half_win = win_size.width / 2;

    for &pt in prev_pts {
        // Extract window from previous image
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let cols_i32 = prev_img.cols() as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let rows_i32 = prev_img.rows() as i32;

        if pt.x < half_win || pt.x >= cols_i32 - half_win
            || pt.y < half_win || pt.y >= rows_i32 - half_win
        {
            next_pts.push(pt);
            status.push(0);
            continue;
        }

        // Simple Lucas-Kanade implementation
        let mut best_match = pt;
        let mut best_error = f64::MAX;

        // Search in a small neighborhood
        let search_range = 10;

        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let next_cols_i32 = next_img.cols() as i32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let next_rows_i32 = next_img.rows() as i32;

        for dy in -search_range..=search_range {
            for dx in -search_range..=search_range {
                let new_pt = Point::new(pt.x + dx, pt.y + dy);

                if new_pt.x < half_win || new_pt.x >= next_cols_i32 - half_win
                    || new_pt.y < half_win || new_pt.y >= next_rows_i32 - half_win
                {
                    continue;
                }

                // Calculate SSD between windows
                let ssd_error = window_ssd(prev_img, next_img, pt, new_pt, half_win)?;

                // Add small distance penalty to prefer points closer to original
                // This prevents drift when multiple points have similar SSD
                let dist = f64::from(dx * dx + dy * dy).sqrt();
                let error = ssd_error + dist * 0.1;

                if error < best_error {
                    best_error = error;
                    best_match = new_pt;
                }
            }
        }

        next_pts.push(best_match);
        status.push(u8::from(best_error < 1000.0));
    }

    Ok((next_pts, status))
}

fn window_ssd(
    img1: &Mat,
    img2: &Mat,
    pt1: Point,
    pt2: Point,
    half_win: i32,
) -> Result<f64> {
    let mut ssd = 0.0;

    for dy in -half_win..=half_win {
        for dx in -half_win..=half_win {
            #[allow(clippy::cast_sign_loss)]
            let y1 = (pt1.y + dy) as usize;
            #[allow(clippy::cast_sign_loss)]
            let x1 = (pt1.x + dx) as usize;
            #[allow(clippy::cast_sign_loss)]
            let y2 = (pt2.y + dy) as usize;
            #[allow(clippy::cast_sign_loss)]
            let x2 = (pt2.x + dx) as usize;

            let val1 = f64::from(img1.at(y1, x1)?[0]);
            let val2 = f64::from(img2.at(y2, x2)?[0]);

            let diff = val1 - val2;
            ssd += diff * diff;
        }
    }

    Ok(ssd)
}

/// Calculate dense optical flow using Farneback method (simplified)
pub fn calc_optical_flow_farneback(
    prev: &Mat,
    next: &Mat,
    pyr_scale: f64,
    levels: i32,
    winsize: i32,
    iterations: i32,
) -> Result<Mat> {
    if prev.channels() != 1 || next.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Farneback requires grayscale images".to_string(),
        ));
    }

    if prev.rows() != next.rows() || prev.cols() != next.cols() {
        return Err(Error::InvalidDimensions(
            "Images must have same dimensions".to_string(),
        ));
    }

    // Create flow matrix (2 channels for x and y flow)
    let mut flow = Mat::new(prev.rows(), prev.cols(), 2, MatDepth::U8)?;

    // Simplified Farneback - calculate gradients
    use crate::imgproc::sobel;

    let mut grad_x = Mat::new(1, 1, 1, MatDepth::U8)?;
    let mut grad_y = Mat::new(1, 1, 1, MatDepth::U8)?;

    sobel(prev, &mut grad_x, 1, 0, 3)?;
    sobel(prev, &mut grad_y, 0, 1, 3)?;

    // For simplicity, use block matching approach
    let block_size = winsize;
    let half_block = block_size / 2;

    #[allow(clippy::cast_sign_loss)]
    let start_row = half_block as usize;
    #[allow(clippy::cast_sign_loss)]
    let end_row = prev.rows() - half_block as usize;
    #[allow(clippy::cast_sign_loss)]
    let start_col = half_block as usize;
    #[allow(clippy::cast_sign_loss)]
    let end_col = prev.cols() - half_block as usize;

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let next_cols_i32 = next.cols() as i32;
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let next_rows_i32 = next.rows() as i32;

    for row in start_row..end_row {
        for col in start_col..end_col {
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let col_i32 = col as i32;
            #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
            let row_i32 = row as i32;
            let pt = Point::new(col_i32, row_i32);

            // Find best match in next image
            let mut best_dx = 0;
            let mut best_dy = 0;
            let mut best_error = f64::MAX;

            for dy in -5..=5 {
                for dx in -5..=5 {
                    let new_pt = Point::new(pt.x + dx, pt.y + dy);

                    if new_pt.x >= half_block && new_pt.x < next_cols_i32 - half_block
                        && new_pt.y >= half_block && new_pt.y < next_rows_i32 - half_block
                    {
                        let error = window_ssd(prev, next, pt, new_pt, half_block)?;

                        if error < best_error {
                            best_error = error;
                            best_dx = dx;
                            best_dy = dy;
                        }
                    }
                }
            }

            let flow_pixel = flow.at_mut(row, col)?;
            #[allow(clippy::cast_possible_truncation)]
            let dx_u8 = (best_dx + 128).clamp(0, 255) as u8;
            #[allow(clippy::cast_possible_truncation)]
            let dy_u8 = (best_dy + 128).clamp(0, 255) as u8;
            flow_pixel[0] = dx_u8;
            flow_pixel[1] = dy_u8;
        }
    }

    Ok(flow)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_optical_flow_lk() {
        let prev = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let next = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(130.0)).unwrap();

        let prev_pts = vec![Point::new(50, 50)];

        let (next_pts, status) = calc_optical_flow_pyr_lk(
            &prev,
            &next,
            &prev_pts,
            Size::new(15, 15),
            2,
        )
        .unwrap();

        assert_eq!(next_pts.len(), 1);
        assert_eq!(status.len(), 1);
    }
}
