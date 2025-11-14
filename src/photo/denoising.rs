use crate::core::Mat;
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
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let y = row as i32 + dy;
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let x = col as i32 + dx;

                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let rows_i32 = src.rows() as i32;
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let cols_i32 = src.cols() as i32;

                    if y >= half_template && y < rows_i32 - half_template
                        && x >= half_template && x < cols_i32 - half_template
                    {
                        // Calculate patch distance
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let row_i32 = row as i32;
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let col_i32 = col as i32;

                        let distance = calculate_color_patch_distance(
                            src,
                            row_i32,
                            col_i32,
                            y,
                            x,
                            half_template,
                        )?;

                        let weight = libm::expf(-distance / (h * h + h_color * h_color));

                        #[allow(clippy::cast_sign_loss)]
                        let y_usize = y as usize;
                        #[allow(clippy::cast_sign_loss)]
                        let x_usize = x as usize;
                        let neighbor = src.at(y_usize, x_usize)?;
                        for ch in 0..3 {
                            pixel_sum[ch] += weight * f32::from(neighbor[ch]);
                        }
                        weight_sum += weight;
                    }
                }
            }

            let result_pixel = result.at_mut(row, col)?;
            #[allow(clippy::cast_possible_truncation)]
            for ch in 0..3 {
                result_pixel[ch] = if weight_sum > 0.0 {
                    let clamped = (pixel_sum[ch] / weight_sum).clamp(0.0, 255.0);
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let pixel_val = clamped as u8;
                    pixel_val
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

            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            let rows_i32 = img.rows() as i32;
            #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
            let cols_i32 = img.cols() as i32;

            if y1 >= 0 && y1 < rows_i32 && x1 >= 0 && x1 < cols_i32
                && y2 >= 0 && y2 < rows_i32 && x2 >= 0 && x2 < cols_i32
            {
                #[allow(clippy::cast_sign_loss)]
                let y1_usize = y1 as usize;
                #[allow(clippy::cast_sign_loss)]
                let x1_usize = x1 as usize;
                #[allow(clippy::cast_sign_loss)]
                let y2_usize = y2 as usize;
                #[allow(clippy::cast_sign_loss)]
                let x2_usize = x2 as usize;

                let p1 = img.at(y1_usize, x1_usize)?;
                let p2 = img.at(y2_usize, x2_usize)?;

                #[allow(clippy::cast_possible_truncation)]
                for ch in 0..3 {
                    let diff = f32::from(p1[ch]) - f32::from(p2[ch]);
                    dist += diff * diff;
                }
                count += 1;
            }
        }
    }

    #[allow(clippy::cast_precision_loss)]
    let count_f32 = count as f32;
    Ok(if count > 0 { dist / count_f32 } else { 0.0 })
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
    #[allow(clippy::cast_sign_loss)]
    let d_usize = d as usize;
    let mut space_weights = vec![vec![0.0f32; d_usize]; d_usize];
    for i in 0..d_usize {
        for j in 0..d_usize {
            #[allow(clippy::cast_precision_loss)]
            let i_f32 = i as f32;
            #[allow(clippy::cast_precision_loss)]
            let j_f32 = j as f32;
            #[allow(clippy::cast_precision_loss)]
            let radius_f32 = radius as f32;
            let dx = i_f32 - radius_f32;
            let dy = j_f32 - radius_f32;
            space_weights[i][j] = libm::expf(-(dx * dx + dy * dy) / (2.0 * sigma_space * sigma_space));
        }
    }

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            #[allow(clippy::cast_possible_truncation)]
            for ch in 0..src.channels() {
                let center_val = f32::from(src.at(row, col)?[ch]);
                let mut sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                for dy in -radius..=radius {
                    for dx in -radius..=radius {
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let row_i32 = row as i32;
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let col_i32 = col as i32;
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let rows_max = src.rows() as i32 - 1;
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let cols_max = src.cols() as i32 - 1;

                        #[allow(clippy::cast_sign_loss)]
                        let y = (row_i32 + dy).clamp(0, rows_max) as usize;
                        #[allow(clippy::cast_sign_loss)]
                        let x = (col_i32 + dx).clamp(0, cols_max) as usize;

                        let neighbor_val = f32::from(src.at(y, x)?[ch]);
                        let color_diff = neighbor_val - center_val;

                        let color_weight = libm::expf(-(color_diff * color_diff) / (2.0 * sigma_color * sigma_color));
                        #[allow(clippy::cast_sign_loss)]
                        let dy_idx = (dy + radius) as usize;
                        #[allow(clippy::cast_sign_loss)]
                        let dx_idx = (dx + radius) as usize;
                        let space_weight = space_weights[dy_idx][dx_idx];

                        let weight = color_weight * space_weight;
                        sum += neighbor_val * weight;
                        weight_sum += weight;
                    }
                }

                let clamped = (sum / weight_sum).clamp(0.0, 255.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let pixel_val = clamped as u8;
                result.at_mut(row, col)?[ch] = pixel_val;
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
                let center = f32::from(result.at(row, col)?[0]);

                // Compute gradients
                let north = f32::from(result.at(row - 1, col)?[0]) - center;
                let south = f32::from(result.at(row + 1, col)?[0]) - center;
                let east = f32::from(result.at(row, col + 1)?[0]) - center;
                let west = f32::from(result.at(row, col - 1)?[0]) - center;

                // Compute conductance (edge-stopping function)
                let cn = libm::expf(-((north / kappa).powi(2)));
                let cs = libm::expf(-((south / kappa).powi(2)));
                let ce = libm::expf(-((east / kappa).powi(2)));
                let cw = libm::expf(-((west / kappa).powi(2)));

                // Update
                let update = lambda * (cn * north + cs * south + ce * east + cw * west);
                let new_val = (center + update).clamp(0.0, 255.0);

                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let pixel_val = new_val as u8;
                next.at_mut(row, col)?[0] = pixel_val;
            }
        }

        result = next;
    }

    Ok(result)
}

/// Median filter for salt-and-pepper noise
pub fn median_filter(src: &Mat, kernel_size: usize) -> Result<Mat> {
    if kernel_size.is_multiple_of(2) {
        return Err(Error::InvalidParameter(
            "Kernel size must be odd".to_string(),
        ));
    }

    let mut result = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;
    let radius = kernel_size / 2;
    let window_size = kernel_size * kernel_size;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            #[allow(clippy::cast_possible_truncation)]
            for ch in 0..src.channels() {
                let mut values = Vec::with_capacity(window_size);

                #[allow(clippy::cast_possible_wrap)]
                let radius_i32 = radius as i32;

                for dy in -radius_i32..=radius_i32 {
                    for dx in -radius_i32..=radius_i32 {
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let row_i32 = row as i32;
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let col_i32 = col as i32;
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let rows_max = src.rows() as i32 - 1;
                        #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                        let cols_max = src.cols() as i32 - 1;

                        #[allow(clippy::cast_sign_loss)]
                        let y = (row_i32 + dy).clamp(0, rows_max) as usize;
                        #[allow(clippy::cast_sign_loss)]
                        let x = (col_i32 + dx).clamp(0, cols_max) as usize;
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

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let row_i32 = row as i32;
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let col_i32 = col as i32;
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let rows_max = src.rows() as i32 - 1;
                    #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                    let cols_max = src.cols() as i32 - 1;

                    #[allow(clippy::cast_sign_loss)]
                    let y = (row_i32 + dy).clamp(0, rows_max) as usize;
                    #[allow(clippy::cast_sign_loss)]
                    let x = (col_i32 + dx).clamp(0, cols_max) as usize;
                    let val = f32::from(src.at(y, x)?[0]);
                    sum += val;
                    sum_sq += val * val;
                    count += 1;
                }
            }

            #[allow(clippy::cast_precision_loss)]
            let count_f32 = count as f32;
            let mean = sum / count_f32;
            let variance = (sum_sq / count_f32) - (mean * mean);

            // Wiener filter formula
            let center = f32::from(src.at(row, col)?[0]);
            let filtered = if variance > noise_variance {
                mean + ((variance - noise_variance) / variance) * (center - mean)
            } else {
                mean
            };

            let clamped = filtered.clamp(0.0, 255.0);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let pixel_val = clamped as u8;
            result.at_mut(row, col)?[0] = pixel_val;
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
                let center = f32::from(result.at(row, col)?[0]);

                // Compute gradients
                let dx_forward = f32::from(result.at(row, col + 1)?[0]) - center;
                let dx_backward = center - f32::from(result.at(row, col - 1)?[0]);
                let dy_forward = f32::from(result.at(row + 1, col)?[0]) - center;
                let dy_backward = center - f32::from(result.at(row - 1, col)?[0]);

                // Compute divergence
                let grad_mag_x = (dx_forward.powi(2) + dy_forward.powi(2)).sqrt() + 1e-8;
                let grad_mag_y = (dx_backward.powi(2) + dy_backward.powi(2)).sqrt() + 1e-8;

                let div = (dx_forward / grad_mag_x - dx_backward / grad_mag_y)
                    + (dy_forward / grad_mag_x - dy_backward / grad_mag_y);

                // Update
                let data_term = center - f32::from(src.at(row, col)?[0]);
                let update = lambda * div - data_term;

                let new_val = (center + 0.1 * update).clamp(0.0, 255.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let pixel_val = new_val as u8;
                next.at_mut(row, col)?[0] = pixel_val;
            }
        }

        result = next;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{MatDepth, types::Scalar};

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
