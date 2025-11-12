use crate::core::{Mat, MatDepth};
use crate::core::types::Size;
use crate::error::{Error, Result};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Apply Gaussian blur to an image
pub fn gaussian_blur(src: &Mat, dst: &mut Mat, ksize: Size, sigma_x: f64) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "gaussian_blur only supports U8 depth".to_string(),
        ));
    }

    if ksize.width % 2 == 0 || ksize.height % 2 == 0 {
        return Err(Error::InvalidParameter(
            "Kernel size must be odd".to_string(),
        ));
    }

    // Try GPU acceleration if available (native only - WASM uses direct GPU bindings)
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    {
        if crate::gpu::gpu_available() && ksize.width == ksize.height {
            if let Ok(()) = crate::gpu::ops::gaussian_blur_gpu(src, dst, ksize, sigma_x) {
                return Ok(());
            }
            // Fall through to CPU on GPU failure
        }
    }

    // CPU implementation (fallback or default)
    gaussian_blur_cpu(src, dst, ksize, sigma_x)
}

/// CPU implementation of Gaussian blur
fn gaussian_blur_cpu(src: &Mat, dst: &mut Mat, ksize: Size, sigma_x: f64) -> Result<()> {
    let kernel = create_gaussian_kernel(ksize, sigma_x)?;
    apply_separable_filter(src, dst, &kernel, &kernel)
}

/// Apply box blur with GPU acceleration (async for WASM)
pub async fn blur_async(src: &Mat, dst: &mut Mat, ksize: Size, use_gpu: bool) -> Result<()> {
    // Try GPU if requested and available
    if use_gpu && ksize.width == ksize.height {
        #[cfg(feature = "gpu")]
        {
            use crate::gpu::ops::box_blur_gpu_async;
            match box_blur_gpu_async(src, dst, ksize.width).await {
                Ok(()) => return Ok(()),
                Err(_) => {
                    // Fall through to CPU
                }
            }
        }
    }

    // CPU fallback
    blur(src, dst, ksize)
}

/// Apply box blur (simple averaging) - optimized with separable filter (CPU-only, sync)
pub fn blur(src: &Mat, dst: &mut Mat, ksize: Size) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "blur only supports U8 depth".to_string(),
        ));
    }

    // Box filter is separable - create uniform kernel
    let kwidth_usize = usize::try_from(ksize.width).unwrap_or(0);
    let kheight_usize = usize::try_from(ksize.height).unwrap_or(0);
    #[allow(clippy::cast_precision_loss)]
    let kernel_x: Vec<f32> = vec![1.0 / ksize.width as f32; kwidth_usize];
    #[allow(clippy::cast_precision_loss)]
    let kernel_y: Vec<f32> = vec![1.0 / ksize.height as f32; kheight_usize];

    apply_separable_filter(src, dst, &kernel_x, &kernel_y)
}

/// Apply median blur - optimized with rayon parallelization
pub fn median_blur(src: &Mat, dst: &mut Mat, ksize: i32) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "median_blur only supports U8 depth".to_string(),
        ));
    }

    if ksize % 2 == 0 {
        return Err(Error::InvalidParameter(
            "Kernel size must be odd".to_string(),
        ));
    }

    if ksize > 21 {
        return Err(Error::InvalidParameter(
            "Kernel size must be <= 21 for median blur".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    let rows = src.rows();
    let cols = src.cols();
    let channels = src.channels();
    let half = ksize / 2;
    let kernel_area = usize::try_from(ksize * ksize).unwrap_or(0);

    // Use rayon::scope to safely share references
    rayon::scope(|_s| {
        let dst_data = dst.data_mut();
        let src_data = src.data();
        let row_size = cols * channels;

        dst_data.par_chunks_mut(row_size).enumerate().for_each(|(row, dst_row)| {
            // Stack array for kernel values (max 21x21 = 441 elements)
            let mut values = [0u8; 441];

            for col in 0..cols {
                for ch in 0..channels {
                    let mut count = 0;

                    // Collect values from kernel window
                    let row_i32 = i32::try_from(row).unwrap_or(i32::MAX);
                    let col_i32 = i32::try_from(col).unwrap_or(i32::MAX);
                    let rows_i32 = i32::try_from(rows).unwrap_or(i32::MAX);
                    let cols_i32 = i32::try_from(cols).unwrap_or(i32::MAX);

                    for ky in -half..=half {
                        let r = usize::try_from((row_i32 + ky).max(0).min(rows_i32 - 1)).unwrap_or(0);
                        for kx in -half..=half {
                            let c = usize::try_from((col_i32 + kx).max(0).min(cols_i32 - 1)).unwrap_or(0);

                            let src_idx = (r * cols + c) * channels + ch;
                            values[count] = src_data[src_idx];
                            count += 1;
                        }
                    }

                    // Sort and find median
                    values[..kernel_area].sort_unstable();
                    let median = values[kernel_area / 2];

                    let dst_idx = col * channels + ch;
                    dst_row[dst_idx] = median;
                }
            }
        });
    });

    Ok(())
}

/// Create a 1D Gaussian kernel
fn create_gaussian_kernel(ksize: Size, sigma: f64) -> Result<Vec<f32>> {
    let size = ksize.width.max(ksize.height);
    if size % 2 == 0 {
        return Err(Error::InvalidParameter(
            "Kernel size must be odd".to_string(),
        ));
    }

    let sigma = if sigma <= 0.0 {
        0.3 * ((f64::from(size) - 1.0) * 0.5 - 1.0) + 0.8
    } else {
        sigma
    };

    let half = size / 2;
    let size_usize = usize::try_from(size).unwrap_or(0);
    let mut kernel = Vec::with_capacity(size_usize);
    let mut sum = 0.0;

    for i in -half..=half {
        let x = f64::from(i);
        let value = (-x * x / (2.0 * sigma * sigma)).exp();
        #[allow(clippy::cast_possible_truncation)]
        kernel.push(value as f32);
        sum += value;
    }

    // Normalize
    #[allow(clippy::cast_possible_truncation)]
    let sum_f32 = sum as f32;
    for val in &mut kernel {
        *val /= sum_f32;
    }

    Ok(kernel)
}

/// Apply separable filter (for efficiency) - parallel version
fn apply_separable_filter(
    src: &Mat,
    dst: &mut Mat,
    kernel_x: &[f32],
    kernel_y: &[f32],
) -> Result<()> {
    let rows = src.rows();
    let cols = src.cols();
    let channels = src.channels();

    // First apply horizontal kernel - PARALLEL
    let mut temp = Mat::new(rows, cols, channels, src.depth())?;

    let half_x = kernel_x.len() / 2;

    // Use rayon::scope to safely share references
    rayon::scope(|s| {
        let temp_data = temp.data_mut();
        let src_data = src.data();

        // Split temp data into rows for parallel processing
        let row_size = cols * channels;

        temp_data.par_chunks_mut(row_size).enumerate().for_each(|(row, temp_row)| {
            let cols_i32 = i32::try_from(cols).unwrap_or(i32::MAX);
            let half_x_i32 = i32::try_from(half_x).unwrap_or(i32::MAX);

            for col in 0..cols {
                let mut sums = [0f32; 4];
                let col_i32 = i32::try_from(col).unwrap_or(i32::MAX);

                for (i, &k) in kernel_x.iter().enumerate() {
                    let i_i32 = i32::try_from(i).unwrap_or(i32::MAX);
                    let offset = i_i32 - half_x_i32;
                    let c = usize::try_from((col_i32 + offset).max(0).min(cols_i32 - 1)).unwrap_or(0);

                    let src_idx = (row * cols + c) * channels;
                    let pixel = &src_data[src_idx..src_idx + channels];

                    match channels {
                        1 => {
                            sums[0] += f32::from(pixel[0]) * k;
                        }
                        3 => {
                            sums[0] += f32::from(pixel[0]) * k;
                            sums[1] += f32::from(pixel[1]) * k;
                            sums[2] += f32::from(pixel[2]) * k;
                        }
                        4 => {
                            sums[0] += f32::from(pixel[0]) * k;
                            sums[1] += f32::from(pixel[1]) * k;
                            sums[2] += f32::from(pixel[2]) * k;
                            sums[3] += f32::from(pixel[3]) * k;
                        }
                        _ => {
                            for ch in 0..channels {
                                sums[ch] += f32::from(pixel[ch]) * k;
                            }
                        }
                    }
                }

                let temp_idx = col * channels;
                let temp_pixel = &mut temp_row[temp_idx..temp_idx + channels];

                match channels {
                    1 => {
                        let clamped = sums[0].round().clamp(0.0, 255.0);
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        { temp_pixel[0] = clamped as u8; }
                    }
                    3 => {
                        let clamped0 = sums[0].round().clamp(0.0, 255.0);
                        let clamped1 = sums[1].round().clamp(0.0, 255.0);
                        let clamped2 = sums[2].round().clamp(0.0, 255.0);
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        {
                            temp_pixel[0] = clamped0 as u8;
                            temp_pixel[1] = clamped1 as u8;
                            temp_pixel[2] = clamped2 as u8;
                        }
                    }
                    4 => {
                        let clamped0 = sums[0].round().clamp(0.0, 255.0);
                        let clamped1 = sums[1].round().clamp(0.0, 255.0);
                        let clamped2 = sums[2].round().clamp(0.0, 255.0);
                        let clamped3 = sums[3].round().clamp(0.0, 255.0);
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        {
                            temp_pixel[0] = clamped0 as u8;
                            temp_pixel[1] = clamped1 as u8;
                            temp_pixel[2] = clamped2 as u8;
                            temp_pixel[3] = clamped3 as u8;
                        }
                    }
                    _ => {
                        for ch in 0..channels {
                            let clamped = sums[ch].round().clamp(0.0, 255.0);
                            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                            { temp_pixel[ch] = clamped as u8; }
                        }
                    }
                }
            }
        });
    });

    // Then apply vertical kernel - PARALLEL
    *dst = Mat::new(rows, cols, channels, src.depth())?;

    let half_y = kernel_y.len() / 2;

    // Vertical pass
    rayon::scope(|s| {
        let dst_data = dst.data_mut();
        let temp_data = temp.data();

        let row_size = cols * channels;

        dst_data.par_chunks_mut(row_size).enumerate().for_each(|(row, dst_row)| {
            let rows_i32 = i32::try_from(rows).unwrap_or(i32::MAX);
            let row_i32 = i32::try_from(row).unwrap_or(i32::MAX);
            let half_y_i32 = i32::try_from(half_y).unwrap_or(i32::MAX);

            for col in 0..cols {
                let mut sums = [0f32; 4];

                for (i, &k) in kernel_y.iter().enumerate() {
                    let i_i32 = i32::try_from(i).unwrap_or(i32::MAX);
                    let offset = i_i32 - half_y_i32;
                    let r = usize::try_from((row_i32 + offset).max(0).min(rows_i32 - 1)).unwrap_or(0);

                    let temp_idx = (r * cols + col) * channels;
                    let pixel = &temp_data[temp_idx..temp_idx + channels];

                    match channels {
                        1 => {
                            sums[0] += f32::from(pixel[0]) * k;
                        }
                        3 => {
                            sums[0] += f32::from(pixel[0]) * k;
                            sums[1] += f32::from(pixel[1]) * k;
                            sums[2] += f32::from(pixel[2]) * k;
                        }
                        4 => {
                            sums[0] += f32::from(pixel[0]) * k;
                            sums[1] += f32::from(pixel[1]) * k;
                            sums[2] += f32::from(pixel[2]) * k;
                            sums[3] += f32::from(pixel[3]) * k;
                        }
                        _ => {
                            for ch in 0..channels {
                                sums[ch] += f32::from(pixel[ch]) * k;
                            }
                        }
                    }
                }

                let dst_idx = col * channels;
                let dst_pixel = &mut dst_row[dst_idx..dst_idx + channels];

                match channels {
                    1 => {
                        let clamped = sums[0].round().clamp(0.0, 255.0);
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        { dst_pixel[0] = clamped as u8; }
                    }
                    3 => {
                        let clamped0 = sums[0].round().clamp(0.0, 255.0);
                        let clamped1 = sums[1].round().clamp(0.0, 255.0);
                        let clamped2 = sums[2].round().clamp(0.0, 255.0);
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        {
                            dst_pixel[0] = clamped0 as u8;
                            dst_pixel[1] = clamped1 as u8;
                            dst_pixel[2] = clamped2 as u8;
                        }
                    }
                    4 => {
                        let clamped0 = sums[0].round().clamp(0.0, 255.0);
                        let clamped1 = sums[1].round().clamp(0.0, 255.0);
                        let clamped2 = sums[2].round().clamp(0.0, 255.0);
                        let clamped3 = sums[3].round().clamp(0.0, 255.0);
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        {
                            dst_pixel[0] = clamped0 as u8;
                            dst_pixel[1] = clamped1 as u8;
                            dst_pixel[2] = clamped2 as u8;
                            dst_pixel[3] = clamped3 as u8;
                        }
                    }
                    _ => {
                        for ch in 0..channels {
                            let clamped = sums[ch].round().clamp(0.0, 255.0);
                            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                            { dst_pixel[ch] = clamped as u8; }
                        }
                    }
                }
            }
        });
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_blur() {
        let src = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        blur(&src, &mut dst, Size::new(5, 5)).unwrap();

        assert_eq!(dst.rows(), src.rows());
        assert_eq!(dst.cols(), src.cols());
    }

    #[test]
    fn test_gaussian_blur() {
        let src = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        gaussian_blur(&src, &mut dst, Size::new(5, 5), 1.0).unwrap();

        assert_eq!(dst.rows(), src.rows());
        assert_eq!(dst.cols(), src.cols());
    }
}
