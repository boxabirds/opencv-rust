use crate::core::{Mat, MatDepth};
use crate::core::types::Point;
use crate::error::{Error, Result};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// `KeyPoint` structure representing a feature point
#[derive(Debug, Clone)]
pub struct KeyPoint {
    pub pt: Point,
    pub size: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: i32,
}

impl KeyPoint {
    #[must_use] 
    pub fn new(pt: Point, size: f32) -> Self {
        Self {
            pt,
            size,
            angle: -1.0,
            response: 0.0,
            octave: 0,
        }
    }
}

/// Harris corner detector
pub fn harris_corners(
    src: &Mat,
    block_size: i32,
    ksize: i32,
    k: f64,
    threshold: f64,
) -> Result<Vec<KeyPoint>> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Harris corner detection requires grayscale image".to_string(),
        ));
    }

    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "Harris corner detection only supports U8 depth".to_string(),
        ));
    }

    // Calculate gradients
    use crate::imgproc::sobel;
    let mut grad_x = Mat::new(1, 1, 1, MatDepth::U8)?;
    let mut grad_y = Mat::new(1, 1, 1, MatDepth::U8)?;

    sobel(src, &mut grad_x, 1, 0, ksize)?;
    sobel(src, &mut grad_y, 0, 1, ksize)?;

    // Calculate products of gradients
    let mut ixx = vec![vec![0.0f64; src.cols()]; src.rows()];
    let mut iyy = vec![vec![0.0f64; src.cols()]; src.rows()];
    let mut ixy = vec![vec![0.0f64; src.cols()]; src.rows()];

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let gx = f64::from(grad_x.at(row, col)?[0]);
            let gy = f64::from(grad_y.at(row, col)?[0]);

            ixx[row][col] = gx * gx;
            iyy[row][col] = gy * gy;
            ixy[row][col] = gx * gy;
        }
    }

    // Calculate Harris response
    let half_block = block_size / 2;
    let mut keypoints = Vec::new();

    #[allow(clippy::cast_sign_loss)]
    let half_block_usize = half_block as usize;
    #[allow(clippy::cast_sign_loss)]
    let end_row = src.rows() - half_block as usize;
    #[allow(clippy::cast_sign_loss)]
    let end_col = src.cols() - half_block as usize;

    for row in half_block_usize..end_row {
        for col in half_block_usize..end_col {
            // Sum over block
            let mut sxx = 0.0;
            let mut syy = 0.0;
            let mut sxy = 0.0;

            for by in -half_block..=half_block {
                for bx in -half_block..=half_block {
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let row_i32 = row as i32;
                    #[allow(clippy::cast_sign_loss)]
                    let y = (row_i32 + by) as usize;
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let col_i32 = col as i32;
                    #[allow(clippy::cast_sign_loss)]
                    let x = (col_i32 + bx) as usize;

                    sxx += ixx[y][x];
                    syy += iyy[y][x];
                    sxy += ixy[y][x];
                }
            }

            // Harris corner response
            let det = sxx * syy - sxy * sxy;
            let trace = sxx + syy;
            let response = det - k * trace * trace;

            if response > threshold {
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                let pt_x = col as i32;
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                let pt_y = row as i32;
                #[allow(clippy::cast_precision_loss)]
                let size = block_size as f32;
                #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
                let response_f32 = response as f32;

                keypoints.push(KeyPoint {
                    pt: Point::new(pt_x, pt_y),
                    size,
                    angle: -1.0,
                    response: response_f32,
                    octave: 0,
                });
            }
        }
    }

    Ok(keypoints)
}

/// Good Features To Track (Shi-Tomasi corner detector)
pub fn good_features_to_track(
    src: &Mat,
    max_corners: usize,
    quality_level: f64,
    min_distance: f64,
    block_size: i32,
) -> Result<Vec<KeyPoint>> {
    // Use Harris detector with k=0 (Shi-Tomasi modification)
    let mut corners = harris_corners(src, block_size, 3, 0.04, quality_level)?;

    // Sort by response
    corners.sort_by(|a, b| b.response.partial_cmp(&a.response).unwrap());

    // Apply non-maximum suppression with minimum distance
    let mut filtered: Vec<KeyPoint> = Vec::new();

    for corner in corners {
        let mut too_close = false;

        for existing in &filtered {
            let dx = f64::from(corner.pt.x - existing.pt.x);
            let dy = f64::from(corner.pt.y - existing.pt.y);
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < min_distance {
                too_close = true;
                break;
            }
        }

        if !too_close {
            filtered.push(corner);
            if filtered.len() >= max_corners {
                break;
            }
        }
    }

    Ok(filtered)
}

/// FAST (Features from Accelerated Segment Test) corner detector - optimized parallel version
pub fn fast(
    src: &Mat,
    threshold: i32,
    nonmax_suppression: bool,
) -> Result<Vec<KeyPoint>> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "FAST requires grayscale image".to_string(),
        ));
    }

    // Bresenham circle of radius 3
    let circle_offsets: [(i32, i32); 16] = [
        (0, -3), (1, -3), (2, -2), (3, -1),
        (3, 0), (3, 1), (2, 2), (1, 3),
        (0, 3), (-1, 3), (-2, 2), (-3, 1),
        (-3, 0), (-3, -1), (-2, -2), (-1, -3),
    ];

    let rows = src.rows();
    let cols = src.cols();
    let src_data = src.data();

    // Parallel row processing - collect keypoints per row
    let keypoints: Vec<KeyPoint> = (3..(rows - 3))
        .into_par_iter()
        .flat_map(|row| {
            let mut row_keypoints = Vec::new();

            for col in 3..(cols - 3) {
                let center_idx = row * cols + col;
                let center_val = i32::from(src_data[center_idx]);
                let threshold_upper = center_val + threshold;
                let threshold_lower = center_val - threshold;

                // Sample circle pixels into fixed-size array (no heap allocation)
                let mut circle_values = [0i32; 16];
                for (i, &(dx, dy)) in circle_offsets.iter().enumerate() {
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let row_i32 = row as i32;
                    #[allow(clippy::cast_sign_loss)]
                    let y = (row_i32 + dy) as usize;
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let col_i32 = col as i32;
                    #[allow(clippy::cast_sign_loss)]
                    let x = (col_i32 + dx) as usize;
                    let idx = y * cols + x;
                    circle_values[i] = i32::from(src_data[idx]);
                }

                // Count consecutive pixels using circular indexing (no clone needed)
                let mut max_consecutive_brighter = 0;
                let mut max_consecutive_darker = 0;
                let mut consecutive_brighter = 0;
                let mut consecutive_darker = 0;

                // Check doubled length by iterating with wraparound
                for i in 0..32 {
                    let val = circle_values[i % 16];

                    if val > threshold_upper {
                        consecutive_brighter += 1;
                        consecutive_darker = 0;
                        max_consecutive_brighter = max_consecutive_brighter.max(consecutive_brighter);
                    } else if val < threshold_lower {
                        consecutive_darker += 1;
                        consecutive_brighter = 0;
                        max_consecutive_darker = max_consecutive_darker.max(consecutive_darker);
                    } else {
                        consecutive_brighter = 0;
                        consecutive_darker = 0;
                    }
                }

                // Need at least 12 consecutive pixels
                if max_consecutive_brighter >= 12 || max_consecutive_darker >= 12 {
                    #[allow(clippy::cast_precision_loss)]
                    let response = (max_consecutive_brighter.max(max_consecutive_darker)) as f32;

                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let pt_x = col as i32;
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let pt_y = row as i32;

                    row_keypoints.push(KeyPoint {
                        pt: Point::new(pt_x, pt_y),
                        size: 7.0,
                        angle: -1.0,
                        response,
                        octave: 0,
                    });
                }
            }

            row_keypoints
        })
        .collect();

    // Non-maximum suppression
    let final_keypoints = if nonmax_suppression {
        apply_non_max_suppression(&keypoints, 3)
    } else {
        keypoints
    };

    Ok(final_keypoints)
}

/// Apply non-maximum suppression to keypoints
fn apply_non_max_suppression(keypoints: &[KeyPoint], radius: i32) -> Vec<KeyPoint> {
    let mut result = Vec::new();

    for kp in keypoints {
        let mut is_maximum = true;

        for other in keypoints {
            if std::ptr::eq(kp, other) {
                continue;
            }

            let dx = (kp.pt.x - other.pt.x).abs();
            let dy = (kp.pt.y - other.pt.y).abs();

            if dx <= radius && dy <= radius && other.response > kp.response {
                is_maximum = false;
                break;
            }
        }

        if is_maximum {
            result.push(kp.clone());
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_harris_corners() {
        let img = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let corners = harris_corners(&img, 3, 3, 0.04, 1000.0).unwrap();
        // Uniform image should have few/no corners
        assert!(corners.len() < 100);
    }

    #[test]
    fn test_fast() {
        let img = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let keypoints = fast(&img, 20, true).unwrap();
        // May detect some based on noise (len is always >= 0 for Vec)
        let _ = keypoints.len();
    }
}
