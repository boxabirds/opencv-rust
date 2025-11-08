use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};

/// Bilateral filter for edge-preserving smoothing
pub fn bilateral_filter(
    src: &Mat,
    dst: &mut Mat,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "bilateral_filter only supports U8 depth".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    let radius = if d <= 0 { 5 } else { d / 2 };

    // Precompute spatial Gaussian kernel
    let mut spatial_kernel = vec![vec![0.0f64; (2 * radius + 1) as usize]; (2 * radius + 1) as usize];
    let space_coeff = -0.5 / (sigma_space * sigma_space);

    for i in -radius..=radius {
        for j in -radius..=radius {
            let dist = (i * i + j * j) as f64;
            spatial_kernel[(i + radius) as usize][(j + radius) as usize] =
                (dist * space_coeff).exp();
        }
    }

    let color_coeff = -0.5 / (sigma_color * sigma_color);

    // Apply bilateral filter
    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let center_pixel = src.at(row, col)?;

            let mut sum = vec![0.0f64; src.channels()];
            let mut weight_sum = 0.0f64;

            for i in -radius..=radius {
                for j in -radius..=radius {
                    let y = (row as i32 + i).max(0).min(src.rows() as i32 - 1) as usize;
                    let x = (col as i32 + j).max(0).min(src.cols() as i32 - 1) as usize;

                    let neighbor_pixel = src.at(y, x)?;

                    // Calculate color distance
                    let mut color_dist = 0.0f64;
                    for ch in 0..src.channels() {
                        let diff = center_pixel[ch] as f64 - neighbor_pixel[ch] as f64;
                        color_dist += diff * diff;
                    }

                    // Combined weight
                    let weight = spatial_kernel[(i + radius) as usize][(j + radius) as usize]
                        * (color_dist * color_coeff).exp();

                    for ch in 0..src.channels() {
                        sum[ch] += neighbor_pixel[ch] as f64 * weight;
                    }
                    weight_sum += weight;
                }
            }

            let dst_pixel = dst.at_mut(row, col)?;
            for ch in 0..src.channels() {
                dst_pixel[ch] = (sum[ch] / weight_sum) as u8;
            }
        }
    }

    Ok(())
}

/// Guided filter for edge-preserving smoothing
pub fn guided_filter(
    src: &Mat,
    guide: &Mat,
    dst: &mut Mat,
    radius: i32,
    eps: f64,
) -> Result<()> {
    if src.rows() != guide.rows() || src.cols() != guide.cols() {
        return Err(Error::InvalidDimensions(
            "Source and guide must have same dimensions".to_string(),
        ));
    }

    if guide.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Guide image must be grayscale".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    // Compute mean of guide image
    let mut mean_I = Mat::new(guide.rows(), guide.cols(), 1, MatDepth::U8)?;
    box_filter(guide, &mut mean_I, radius)?;

    // Compute mean of source image
    let mut mean_p = Mat::new(src.rows(), src.cols(), src.channels(), MatDepth::U8)?;
    box_filter(src, &mut mean_p, radius)?;

    // Compute correlation of guide
    let mut corr_I = Mat::new(guide.rows(), guide.cols(), 1, MatDepth::U8)?;
    for row in 0..guide.rows() {
        for col in 0..guide.cols() {
            let val = guide.at(row, col)?[0] as f64;
            let corr_pixel = corr_I.at_mut(row, col)?;
            corr_pixel[0] = ((val * val) / 255.0) as u8;
        }
    }
    let mut mean_II = Mat::new(guide.rows(), guide.cols(), 1, MatDepth::U8)?;
    box_filter(&corr_I, &mut mean_II, radius)?;

    // Compute variance
    let mut var_I = Mat::new(guide.rows(), guide.cols(), 1, MatDepth::U8)?;
    for row in 0..guide.rows() {
        for col in 0..guide.cols() {
            let mean_val = mean_I.at(row, col)?[0] as f64;
            let mean_sq = mean_II.at(row, col)?[0] as f64;
            let variance = mean_sq - (mean_val * mean_val) / 255.0;

            let var_pixel = var_I.at_mut(row, col)?;
            var_pixel[0] = variance.max(0.0) as u8;
        }
    }

    // Compute coefficients a and b
    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let var = var_I.at(row, col)?[0] as f64;
            let mean_guide = mean_I.at(row, col)?[0] as f64;

            let a = var / (var + eps);
            let mean_src = mean_p.at(row, col)?[0] as f64;
            let b = mean_src - a * mean_guide;

            // Apply linear transform
            let guide_val = guide.at(row, col)?[0] as f64;
            let result = a * guide_val + b;

            let dst_pixel = dst.at_mut(row, col)?;
            for ch in 0..src.channels() {
                dst_pixel[ch] = result.max(0.0).min(255.0) as u8;
            }
        }
    }

    Ok(())
}

fn box_filter(src: &Mat, dst: &mut Mat, radius: i32) -> Result<()> {
    use crate::imgproc::blur;
    use crate::core::types::Size;

    let ksize = 2 * radius + 1;
    blur(src, dst, Size::new(ksize, ksize))
}

/// Distance transform using chamfer distance
pub fn distance_transform(
    src: &Mat,
    dst: &mut Mat,
    distance_type: DistanceType,
    mask_size: i32,
) -> Result<()> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "distance_transform requires single-channel image".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    // Initialize distance map
    let mut dist = vec![vec![f32::MAX; src.cols()]; src.rows()];

    // Set zero distance for foreground pixels
    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let pixel = src.at(row, col)?;
            if pixel[0] > 0 {
                dist[row][col] = 0.0;
            }
        }
    }

    // Forward pass
    for row in 1..src.rows() {
        for col in 1..src.cols() {
            if dist[row][col] > 0.0 {
                let d1 = dist[row - 1][col] + 1.0;
                let d2 = dist[row][col - 1] + 1.0;
                let d3 = dist[row - 1][col - 1] + 1.414;

                dist[row][col] = dist[row][col].min(d1).min(d2).min(d3);
            }
        }
    }

    // Backward pass
    for row in (0..src.rows() - 1).rev() {
        for col in (0..src.cols() - 1).rev() {
            if dist[row][col] > 0.0 {
                let d1 = dist[row + 1][col] + 1.0;
                let d2 = dist[row][col + 1] + 1.0;
                let d3 = dist[row + 1][col + 1] + 1.414;

                dist[row][col] = dist[row][col].min(d1).min(d2).min(d3);
            }
        }
    }

    // Normalize to 0-255
    let max_dist = dist.iter()
        .flat_map(|row| row.iter())
        .fold(0.0f32, |a, &b| a.max(b));

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let dst_pixel = dst.at_mut(row, col)?;
            if max_dist > 0.0 {
                dst_pixel[0] = ((dist[row][col] / max_dist) * 255.0) as u8;
            } else {
                dst_pixel[0] = 0;
            }
        }
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub enum DistanceType {
    L1,
    L2,
    C,
}

/// Watershed segmentation algorithm
pub fn watershed(image: &Mat, markers: &mut Mat) -> Result<()> {
    if image.channels() != 3 || markers.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Watershed requires 3-channel image and 1-channel markers".to_string(),
        ));
    }

    if image.rows() != markers.rows() || image.cols() != markers.cols() {
        return Err(Error::InvalidDimensions(
            "Image and markers must have same size".to_string(),
        ));
    }

    // Compute gradient magnitude
    use crate::imgproc::sobel;
    use crate::imgproc::cvt_color;
    use crate::core::types::ColorConversionCode;

    let mut gray = Mat::new(1, 1, 1, MatDepth::U8)?;
    cvt_color(image, &mut gray, ColorConversionCode::RgbToGray)?;

    let mut grad_x = Mat::new(1, 1, 1, MatDepth::U8)?;
    let mut grad_y = Mat::new(1, 1, 1, MatDepth::U8)?;
    sobel(&gray, &mut grad_x, 1, 0, 3)?;
    sobel(&gray, &mut grad_y, 0, 1, 3)?;

    let mut gradient = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)?;
    for row in 0..gray.rows() {
        for col in 0..gray.cols() {
            let gx = grad_x.at(row, col)?[0] as f32;
            let gy = grad_y.at(row, col)?[0] as f32;
            let mag = (gx * gx + gy * gy).sqrt();

            let grad_pixel = gradient.at_mut(row, col)?;
            grad_pixel[0] = mag.min(255.0) as u8;
        }
    }

    // Priority queue for watershed
    let mut queue: Vec<(u8, usize, usize)> = Vec::new();

    // Initialize queue with marker boundaries
    for row in 1..markers.rows() - 1 {
        for col in 1..markers.cols() - 1 {
            let label = markers.at(row, col)?[0];

            if label > 0 {
                // Check if on boundary
                let mut is_boundary = false;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dy == 0 && dx == 0 { continue; }

                        let ny = (row as i32 + dy) as usize;
                        let nx = (col as i32 + dx) as usize;

                        let neighbor_label = markers.at(ny, nx)?[0];
                        if neighbor_label == 0 {
                            is_boundary = true;
                            break;
                        }
                    }
                    if is_boundary { break; }
                }

                if is_boundary {
                    let grad_val = gradient.at(row, col)?[0];
                    queue.push((grad_val, row, col));
                }
            }
        }
    }

    // Sort by gradient (priority queue simulation)
    queue.sort_by_key(|(grad, _, _)| *grad);

    // Flood fill from markers
    while let Some((_grad, row, col)) = queue.pop() {
        let current_label = markers.at(row, col)?[0];

        for dy in -1..=1 {
            for dx in -1..=1 {
                if dy == 0 && dx == 0 { continue; }

                let ny = row as i32 + dy;
                let nx = col as i32 + dx;

                if ny >= 0 && ny < markers.rows() as i32 && nx >= 0 && nx < markers.cols() as i32 {
                    let ny = ny as usize;
                    let nx = nx as usize;

                    let neighbor_label = markers.at(ny, nx)?[0];

                    if neighbor_label == 0 {
                        // Assign label
                        let neighbor_pixel = markers.at_mut(ny, nx)?;
                        neighbor_pixel[0] = current_label;

                        // Add to queue
                        let grad_val = gradient.at(ny, nx)?[0];
                        queue.push((grad_val, ny, nx));
                    }
                }
            }
        }

        queue.sort_by_key(|(grad, _, _)| *grad);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_bilateral_filter() {
        let src = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        bilateral_filter(&src, &mut dst, 5, 50.0, 50.0).unwrap();
        assert_eq!(dst.rows(), src.rows());
    }

    #[test]
    fn test_distance_transform() {
        let src = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(255.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        distance_transform(&src, &mut dst, DistanceType::L2, 3).unwrap();
        assert_eq!(dst.rows(), src.rows());
    }
}
