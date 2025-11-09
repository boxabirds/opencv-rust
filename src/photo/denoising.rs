use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};

/// Non-local Means Denoising for color images
pub fn fast_nl_means_denoising_colored(
    src: &Mat,
    h: f32,
    h_color: f32,
    template_window_size: i32,
    search_window_size: i32,
) -> Result<Mat> {
    if src.channels() != 3 {
        return Err(Error::InvalidParameter(
            "Color denoising requires 3-channel image".to_string(),
        ));
    }

    let mut result = Mat::new(src.rows(), src.cols(), 3, src.depth())?;

    let half_template = template_window_size / 2;
    let half_search = search_window_size / 2;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let mut weight_sum = 0.0f32;
            let mut pixel_sum = [0.0f32; 3];

            // Search in neighborhood
            for dy in -half_search..=half_search {
                for dx in -half_search..=half_search {
                    let y = row as i32 + dy;
                    let x = col as i32 + dx;

                    if y >= half_template && y < src.rows() as i32 - half_template
                        && x >= half_template && x < src.cols() as i32 - half_template
                    {
                        // Calculate patch distance
                        let distance = calculate_color_patch_distance(
                            src,
                            row as i32,
                            col as i32,
                            y,
                            x,
                            half_template,
                        )?;

                        let weight = (-distance / (h * h + h_color * h_color)).exp();

                        let neighbor = src.at(y as usize, x as usize)?;
                        for ch in 0..3 {
                            pixel_sum[ch] += weight * neighbor[ch] as f32;
                        }
                        weight_sum += weight;
                    }
                }
            }

            let result_pixel = result.at_mut(row, col)?;
            for ch in 0..3 {
                result_pixel[ch] = if weight_sum > 0.0 {
                    (pixel_sum[ch] / weight_sum) as u8
                } else {
                    src.at(row, col)?[ch]
                };
            }
        }
    }

    Ok(result)
}

fn calculate_color_patch_distance(
    img: &Mat,
    row1: i32,
    col1: i32,
    row2: i32,
    col2: i32,
    half_size: i32,
) -> Result<f32> {
    let mut dist = 0.0f32;
    let mut count = 0;

    for dy in -half_size..=half_size {
        for dx in -half_size..=half_size {
            let y1 = row1 + dy;
            let x1 = col1 + dx;
            let y2 = row2 + dy;
            let x2 = col2 + dx;

            if y1 >= 0 && y1 < img.rows() as i32 && x1 >= 0 && x1 < img.cols() as i32
                && y2 >= 0 && y2 < img.rows() as i32 && x2 >= 0 && x2 < img.cols() as i32
            {
                let p1 = img.at(y1 as usize, x1 as usize)?;
                let p2 = img.at(y2 as usize, x2 as usize)?;

                for ch in 0..3 {
                    let diff = p1[ch] as f32 - p2[ch] as f32;
                    dist += diff * diff;
                }
                count += 1;
            }
        }
    }

    Ok(if count > 0 { dist / count as f32 } else { 0.0 })
}

/// Bilateral filter for edge-preserving smoothing
pub fn bilateral_filter(
    src: &Mat,
    d: i32,
    sigma_color: f32,
    sigma_space: f32,
) -> Result<Mat> {
    let mut result = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    let radius = d / 2;

    // Precompute spatial Gaussian weights
    let mut space_weights = vec![vec![0.0f32; d as usize]; d as usize];
    for i in 0..d as usize {
        for j in 0..d as usize {
            let dx = i as f32 - radius as f32;
            let dy = j as f32 - radius as f32;
            space_weights[i][j] = (-(dx * dx + dy * dy) / (2.0 * sigma_space * sigma_space)).exp();
        }
    }

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            for ch in 0..src.channels() {
                let center_val = src.at(row, col)?[ch] as f32;
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        let y = (row as i32 + dy).clamp(0, src.rows() as i32 - 1) as usize;
                        let x = (col as i32 + dx).clamp(0, src.cols() as i32 - 1) as usize;

                        let neighbor_val = src.at(y, x)?[ch] as f32;
                        let color_diff = neighbor_val - center_val;

                        let color_weight = (-(color_diff * color_diff) / (2.0 * sigma_color * sigma_color)).exp();
                        let space_weight = space_weights[(dy + radius) as usize][(dx + radius) as usize];

                        let weight = color_weight * space_weight;
                        sum += neighbor_val * weight;
                        weight_sum += weight;
                    }
                }

                result.at_mut(row, col)?[ch] = (sum / weight_sum) as u8;
            }
        }
    }

    Ok(result)
}

/// Anisotropic diffusion (Perona-Malik)
pub fn anisotropic_diffusion(
    src: &Mat,
    iterations: usize,
    kappa: f32,
    lambda: f32,
) -> Result<Mat> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Anisotropic diffusion requires grayscale image".to_string(),
        ));
    }

    let mut result = src.clone_mat();

    for _ in 0..iterations {
        let mut next = Mat::new(src.rows(), src.cols(), 1, src.depth())?;

        for row in 1..src.rows() - 1 {
            for col in 1..src.cols() - 1 {
                let center = result.at(row, col)?[0] as f32;

                // Compute gradients
                let north = result.at(row - 1, col)?[0] as f32 - center;
                let south = result.at(row + 1, col)?[0] as f32 - center;
                let east = result.at(row, col + 1)?[0] as f32 - center;
                let west = result.at(row, col - 1)?[0] as f32 - center;

                // Compute conductance (edge-stopping function)
                let cn = (-((north / kappa).powi(2))).exp();
                let cs = (-((south / kappa).powi(2))).exp();
                let ce = (-((east / kappa).powi(2))).exp();
                let cw = (-((west / kappa).powi(2))).exp();

                // Update
                let update = lambda * (cn * north + cs * south + ce * east + cw * west);
                let new_val = (center + update).clamp(0.0, 255.0);

                next.at_mut(row, col)?[0] = new_val as u8;
            }
        }

        result = next;
    }

    Ok(result)
}

/// Median filter for salt-and-pepper noise
pub fn median_filter(src: &Mat, kernel_size: usize) -> Result<Mat> {
    if kernel_size % 2 == 0 {
        return Err(Error::InvalidParameter(
            "Kernel size must be odd".to_string(),
        ));
    }

    let mut result = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;
    let radius = kernel_size / 2;
    let window_size = kernel_size * kernel_size;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            for ch in 0..src.channels() {
                let mut values = Vec::with_capacity(window_size);

                for dy in -(radius as i32)..=(radius as i32) {
                    for dx in -(radius as i32)..=(radius as i32) {
                        let y = (row as i32 + dy).clamp(0, src.rows() as i32 - 1) as usize;
                        let x = (col as i32 + dx).clamp(0, src.cols() as i32 - 1) as usize;
                        values.push(src.at(y, x)?[ch]);
                    }
                }

                values.sort_unstable();
                result.at_mut(row, col)?[ch] = values[values.len() / 2];
            }
        }
    }

    Ok(result)
}

/// Wiener filter for noise reduction
pub fn wiener_filter(src: &Mat, noise_variance: f32) -> Result<Mat> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Wiener filter requires grayscale image".to_string(),
        ));
    }

    let mut result = Mat::new(src.rows(), src.cols(), 1, src.depth())?;
    let window_size = 5;
    let radius = window_size / 2;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            // Compute local statistics
            let mut sum = 0.0f32;
            let mut sum_sq = 0.0f32;
            let mut count = 0;

            for dy in -(radius as i32)..=(radius as i32) {
                for dx in -(radius as i32)..=(radius as i32) {
                    let y = (row as i32 + dy).clamp(0, src.rows() as i32 - 1) as usize;
                    let x = (col as i32 + dx).clamp(0, src.cols() as i32 - 1) as usize;
                    let val = src.at(y, x)?[0] as f32;
                    sum += val;
                    sum_sq += val * val;
                    count += 1;
                }
            }

            let mean = sum / count as f32;
            let variance = (sum_sq / count as f32) - (mean * mean);

            // Wiener filter formula
            let center = src.at(row, col)?[0] as f32;
            let filtered = if variance > noise_variance {
                mean + ((variance - noise_variance) / variance) * (center - mean)
            } else {
                mean
            };

            result.at_mut(row, col)?[0] = filtered.clamp(0.0, 255.0) as u8;
        }
    }

    Ok(result)
}

/// Total Variation denoising
pub fn total_variation_denoise(
    src: &Mat,
    lambda: f32,
    iterations: usize,
) -> Result<Mat> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "TV denoising requires grayscale image".to_string(),
        ));
    }

    let mut result = src.clone_mat();

    for _ in 0..iterations {
        let mut next = Mat::new(src.rows(), src.cols(), 1, src.depth())?;

        for row in 1..src.rows() - 1 {
            for col in 1..src.cols() - 1 {
                let center = result.at(row, col)?[0] as f32;

                // Compute gradients
                let dx_forward = result.at(row, col + 1)?[0] as f32 - center;
                let dx_backward = center - result.at(row, col - 1)?[0] as f32;
                let dy_forward = result.at(row + 1, col)?[0] as f32 - center;
                let dy_backward = center - result.at(row - 1, col)?[0] as f32;

                // Compute divergence
                let grad_mag_x = (dx_forward.powi(2) + dy_forward.powi(2)).sqrt() + 1e-8;
                let grad_mag_y = (dx_backward.powi(2) + dy_backward.powi(2)).sqrt() + 1e-8;

                let div = (dx_forward / grad_mag_x - dx_backward / grad_mag_y)
                    + (dy_forward / grad_mag_x - dy_backward / grad_mag_y);

                // Update
                let data_term = center - src.at(row, col)?[0] as f32;
                let update = lambda * div - data_term;

                let new_val = (center + 0.1 * update).clamp(0.0, 255.0);
                next.at_mut(row, col)?[0] = new_val as u8;
            }
        }

        result = next;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_nl_means_colored() {
        let src = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let result = fast_nl_means_denoising_colored(&src, 10.0, 10.0, 7, 21).unwrap();
        assert_eq!(result.rows(), 50);
        assert_eq!(result.channels(), 3);
    }

    #[test]
    fn test_bilateral_filter() {
        let src = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let result = bilateral_filter(&src, 9, 75.0, 75.0).unwrap();
        assert_eq!(result.rows(), 50);
    }

    #[test]
    fn test_anisotropic_diffusion() {
        let src = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let result = anisotropic_diffusion(&src, 5, 10.0, 0.25).unwrap();
        assert_eq!(result.rows(), 50);
    }

    #[test]
    fn test_median_filter() {
        let src = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let result = median_filter(&src, 5).unwrap();
        assert_eq!(result.rows(), 50);
    }

    #[test]
    fn test_wiener_filter() {
        let src = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let result = wiener_filter(&src, 100.0).unwrap();
        assert_eq!(result.rows(), 50);
    }

    #[test]
    fn test_tv_denoise() {
        let src = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        let result = total_variation_denoise(&src, 0.1, 10).unwrap();
        assert_eq!(result.rows(), 50);
    }
}
