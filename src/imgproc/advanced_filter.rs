use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Bilateral filter for edge-preserving smoothing - optimized with rayon parallelization
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

    let rows = src.rows();
    let cols = src.cols();
    let channels = src.channels();
    let radius = if d <= 0 { 5 } else { d / 2 };

    // Precompute spatial Gaussian kernel
    let mut spatial_kernel = vec![vec![0.0f64; (2 * radius + 1) as usize]; (2 * radius + 1) as usize];
    let space_coeff = -0.5 / (sigma_space * sigma_space);

    for i in -radius..=radius {
        for j in -radius..=radius {
            let dist = f64::from(i * i + j * j);
            spatial_kernel[(i + radius) as usize][(j + radius) as usize] =
                (dist * space_coeff).exp();
        }
    }

    let color_coeff = -0.5 / (sigma_color * sigma_color);

    // Use rayon::scope to safely share references
    rayon::scope(|_s| {
        let dst_data = dst.data_mut();
        let src_data = src.data();
        let row_size = cols * channels;

        dst_data.par_chunks_mut(row_size).enumerate().for_each(|(row, dst_row)| {
            // Stack arrays for temporary storage (max 4 channels)
            let mut sum = [0.0f64; 4];
            let mut center = [0u8; 4];

            for col in 0..cols {
                // Get center pixel
                let center_idx = (row * cols + col) * channels;
                for ch in 0..channels {
                    center[ch] = src_data[center_idx + ch];
                }

                sum.fill(0.0);
                let mut weight_sum = 0.0f64;

                // Process neighborhood
                for i in -radius..=radius {
                    let y = (row as i32 + i).max(0).min(rows as i32 - 1) as usize;
                    for j in -radius..=radius {
                        let x = (col as i32 + j).max(0).min(cols as i32 - 1) as usize;

                        let neighbor_idx = (y * cols + x) * channels;

                        // Calculate color distance
                        let mut color_dist = 0.0f64;
                        for ch in 0..channels {
                            let diff = f64::from(center[ch]) - f64::from(src_data[neighbor_idx + ch]);
                            color_dist += diff * diff;
                        }

                        // Combined weight
                        let weight = spatial_kernel[(i + radius) as usize][(j + radius) as usize]
                            * (color_dist * color_coeff).exp();

                        for ch in 0..channels {
                            sum[ch] += f64::from(src_data[neighbor_idx + ch]) * weight;
                        }
                        weight_sum += weight;
                    }
                }

                // Write result
                let dst_idx = col * channels;
                for ch in 0..channels {
                    dst_row[dst_idx + ch] = (sum[ch] / weight_sum) as u8;
                }
            }
        });
    });

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
    let mut mean_i = Mat::new(guide.rows(), guide.cols(), 1, MatDepth::U8)?;
    box_filter(guide, &mut mean_i, radius)?;

    // Compute mean of source image
    let mut mean_p = Mat::new(src.rows(), src.cols(), src.channels(), MatDepth::U8)?;
    box_filter(src, &mut mean_p, radius)?;

    // Compute correlation of guide
    let mut corr_i = Mat::new(guide.rows(), guide.cols(), 1, MatDepth::U8)?;
    for row in 0..guide.rows() {
        for col in 0..guide.cols() {
            let val = f64::from(guide.at(row, col)?[0]);
            let corr_pixel = corr_i.at_mut(row, col)?;
            corr_pixel[0] = ((val * val) / 255.0) as u8;
        }
    }
    let mut mean_ii = Mat::new(guide.rows(), guide.cols(), 1, MatDepth::U8)?;
    box_filter(&corr_i, &mut mean_ii, radius)?;

    // Compute variance
    let mut var_i = Mat::new(guide.rows(), guide.cols(), 1, MatDepth::U8)?;
    for row in 0..guide.rows() {
        for col in 0..guide.cols() {
            let mean_val = f64::from(mean_i.at(row, col)?[0]);
            let mean_sq = f64::from(mean_ii.at(row, col)?[0]);
            let variance = mean_sq - (mean_val * mean_val) / 255.0;

            let var_pixel = var_i.at_mut(row, col)?;
            var_pixel[0] = variance.max(0.0) as u8;
        }
    }

    // Compute coefficients a and b
    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let var = f64::from(var_i.at(row, col)?[0]);
            let mean_guide = f64::from(mean_i.at(row, col)?[0]);

            let a = var / (var + eps);
            let mean_src = f64::from(mean_p.at(row, col)?[0]);
            let b = mean_src - a * mean_guide;

            // Apply linear transform
            let guide_val = f64::from(guide.at(row, col)?[0]);
            let result = a * guide_val + b;

            let dst_pixel = dst.at_mut(row, col)?;
            for ch in 0..src.channels() {
                dst_pixel[ch] = result.clamp(0.0, 255.0) as u8;
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
            let gx = f32::from(grad_x.at(row, col)?[0]);
            let gy = f32::from(grad_y.at(row, col)?[0]);
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

/// Gabor filter for texture analysis and feature extraction
pub fn gabor_filter(
    src: &Mat,
    dst: &mut Mat,
    ksize: i32,
    sigma: f64,
    theta: f64,
    lambda: f64,
    gamma: f64,
    psi: f64,
) -> Result<()> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "gabor_filter requires single-channel image".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    // Generate Gabor kernel
    let kernel = generate_gabor_kernel(ksize, sigma, theta, lambda, gamma, psi);

    // Apply convolution
    let half = ksize / 2;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let mut sum = 0.0f64;

            for ky in -half..=half {
                for kx in -half..=half {
                    let y = (row as i32 + ky).max(0).min(src.rows() as i32 - 1) as usize;
                    let x = (col as i32 + kx).max(0).min(src.cols() as i32 - 1) as usize;

                    let pixel = f64::from(src.at(y, x)?[0]);
                    let k_val = kernel[(ky + half) as usize][(kx + half) as usize];

                    sum += pixel * k_val;
                }
            }

            let dst_pixel = dst.at_mut(row, col)?;
            dst_pixel[0] = sum.abs().min(255.0) as u8;
        }
    }

    Ok(())
}

fn generate_gabor_kernel(
    ksize: i32,
    sigma: f64,
    theta: f64,
    lambda: f64,
    gamma: f64,
    psi: f64,
) -> Vec<Vec<f64>> {
    let half = ksize / 2;
    let mut kernel = vec![vec![0.0; ksize as usize]; ksize as usize];

    let sigma_x = sigma;
    let sigma_y = sigma / gamma;

    for y in -half..=half {
        for x in -half..=half {
            let x_theta = f64::from(x) * theta.cos() + f64::from(y) * theta.sin();
            let y_theta = f64::from(-x) * theta.sin() + f64::from(y) * theta.cos();

            let gaussian = (-(x_theta * x_theta / (2.0 * sigma_x * sigma_x)
                + y_theta * y_theta / (2.0 * sigma_y * sigma_y)))
                .exp();

            let sinusoid = (2.0 * std::f64::consts::PI * x_theta / lambda + psi).cos();

            kernel[(y + half) as usize][(x + half) as usize] = gaussian * sinusoid;
        }
    }

    kernel
}

/// Laplacian of Gaussian (`LoG`) filter for blob detection
pub fn laplacian_of_gaussian(
    src: &Mat,
    dst: &mut Mat,
    ksize: i32,
    sigma: f64,
) -> Result<()> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "laplacian_of_gaussian requires single-channel image".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    // Generate LoG kernel
    let kernel = generate_log_kernel(ksize, sigma);

    // Apply convolution
    let half = ksize / 2;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let mut sum = 0.0f64;

            for ky in -half..=half {
                for kx in -half..=half {
                    let y = (row as i32 + ky).max(0).min(src.rows() as i32 - 1) as usize;
                    let x = (col as i32 + kx).max(0).min(src.cols() as i32 - 1) as usize;

                    let pixel = f64::from(src.at(y, x)?[0]);
                    let k_val = kernel[(ky + half) as usize][(kx + half) as usize];

                    sum += pixel * k_val;
                }
            }

            let dst_pixel = dst.at_mut(row, col)?;
            // Zero-crossing or absolute value
            dst_pixel[0] = sum.abs().min(255.0) as u8;
        }
    }

    Ok(())
}

fn generate_log_kernel(ksize: i32, sigma: f64) -> Vec<Vec<f64>> {
    let half = ksize / 2;
    let mut kernel = vec![vec![0.0; ksize as usize]; ksize as usize];

    let sigma2 = sigma * sigma;
    let sigma4 = sigma2 * sigma2;

    for y in -half..=half {
        for x in -half..=half {
            let x2 = f64::from(x * x);
            let y2 = f64::from(y * y);
            let r2 = x2 + y2;

            // LoG formula: -1/(π*σ^4) * (1 - r²/(2σ²)) * exp(-r²/(2σ²))
            let gaussian = (-r2 / (2.0 * sigma2)).exp();
            let laplacian = (1.0 - r2 / (2.0 * sigma2)) * gaussian;
            kernel[(y + half) as usize][(x + half) as usize] =
                -laplacian / (std::f64::consts::PI * sigma4);
        }
    }

    kernel
}

/// Non-local means denoising
pub fn non_local_means_denoising(
    src: &Mat,
    dst: &mut Mat,
    h: f32,
    template_window_size: i32,
    search_window_size: i32,
) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "non_local_means_denoising only supports U8 depth".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    let t_half = template_window_size / 2;
    let s_half = search_window_size / 2;
    let h2 = h * h;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let mut sum = vec![0.0f32; src.channels()];
            let mut weight_sum = 0.0f32;

            // Search window
            for sy in -s_half..=s_half {
                for sx in -s_half..=s_half {
                    let search_row = (row as i32 + sy).max(0).min(src.rows() as i32 - 1) as usize;
                    let search_col = (col as i32 + sx).max(0).min(src.cols() as i32 - 1) as usize;

                    // Compute patch distance
                    let mut patch_dist = 0.0f32;
                    let mut patch_count = 0;

                    for ty in -t_half..=t_half {
                        for tx in -t_half..=t_half {
                            let r1 = (row as i32 + ty).max(0).min(src.rows() as i32 - 1) as usize;
                            let c1 = (col as i32 + tx).max(0).min(src.cols() as i32 - 1) as usize;

                            let r2 = (search_row as i32 + ty).max(0).min(src.rows() as i32 - 1) as usize;
                            let c2 = (search_col as i32 + tx).max(0).min(src.cols() as i32 - 1) as usize;

                            let p1 = src.at(r1, c1)?;
                            let p2 = src.at(r2, c2)?;

                            for ch in 0..src.channels() {
                                let diff = f32::from(p1[ch]) - f32::from(p2[ch]);
                                patch_dist += diff * diff;
                            }
                            patch_count += 1;
                        }
                    }

                    patch_dist /= patch_count as f32;

                    // Compute weight
                    let weight = (-patch_dist / h2).exp();

                    let search_pixel = src.at(search_row, search_col)?;
                    for ch in 0..src.channels() {
                        sum[ch] += f32::from(search_pixel[ch]) * weight;
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

/// Anisotropic diffusion (Perona-Malik)
pub fn anisotropic_diffusion(
    src: &Mat,
    dst: &mut Mat,
    num_iterations: usize,
    kappa: f32,
    lambda: f32,
) -> Result<()> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "anisotropic_diffusion requires single-channel image".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    // Copy source to dst
    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let pixel = src.at(row, col)?[0];
            dst.at_mut(row, col)?[0] = pixel;
        }
    }

    let kappa2 = kappa * kappa;

    for _ in 0..num_iterations {
        let current = dst.clone();

        for row in 1..dst.rows() - 1 {
            for col in 1..dst.cols() - 1 {
                let center = f32::from(current.at(row, col)?[0]);

                // Compute gradients to neighbors
                let n = f32::from(current.at(row - 1, col)?[0]);
                let s = f32::from(current.at(row + 1, col)?[0]);
                let e = f32::from(current.at(row, col + 1)?[0]);
                let w = f32::from(current.at(row, col - 1)?[0]);

                let grad_n = n - center;
                let grad_s = s - center;
                let grad_e = e - center;
                let grad_w = w - center;

                // Compute diffusion coefficients (Perona-Malik type 2)
                let c_n = 1.0 / (1.0 + (grad_n * grad_n) / kappa2);
                let c_s = 1.0 / (1.0 + (grad_s * grad_s) / kappa2);
                let c_e = 1.0 / (1.0 + (grad_e * grad_e) / kappa2);
                let c_w = 1.0 / (1.0 + (grad_w * grad_w) / kappa2);

                // Update pixel
                let update = lambda * (c_n * grad_n + c_s * grad_s + c_e * grad_e + c_w * grad_w);
                let new_val = center + update;

                dst.at_mut(row, col)?[0] = new_val.clamp(0.0, 255.0) as u8;
            }
        }
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

    #[test]
    fn test_gabor_filter() {
        let src = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        gabor_filter(&src, &mut dst, 21, 5.0, 0.0, 10.0, 0.5, 0.0).unwrap();
        assert_eq!(dst.rows(), src.rows());
    }

    #[test]
    fn test_laplacian_of_gaussian() {
        let src = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        laplacian_of_gaussian(&src, &mut dst, 9, 1.5).unwrap();
        assert_eq!(dst.rows(), src.rows());
    }

    #[test]
    fn test_non_local_means() {
        let src = Mat::new_with_default(30, 30, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        non_local_means_denoising(&src, &mut dst, 10.0, 3, 7).unwrap();
        assert_eq!(dst.rows(), src.rows());
    }

    #[test]
    fn test_anisotropic_diffusion() {
        let src = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        anisotropic_diffusion(&src, &mut dst, 5, 10.0, 0.25).unwrap();
        assert_eq!(dst.rows(), src.rows());
    }
}
