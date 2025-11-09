use crate::core::{Mat, MatDepth};
use crate::core::types::Size;
use crate::error::{Error, Result};
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

    let kernel = create_gaussian_kernel(ksize, sigma_x)?;
    apply_separable_filter(src, dst, &kernel, &kernel)
}

/// Apply box blur (simple averaging) - optimized with separable filter
pub fn blur(src: &Mat, dst: &mut Mat, ksize: Size) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "blur only supports U8 depth".to_string(),
        ));
    }

    // Box filter is separable - create uniform kernel
    let kernel_x: Vec<f32> = vec![1.0 / ksize.width as f32; ksize.width as usize];
    let kernel_y: Vec<f32> = vec![1.0 / ksize.height as f32; ksize.height as usize];

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
    let kernel_area = (ksize * ksize) as usize;

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
                    for ky in -half..=half {
                        let r = (row as i32 + ky).max(0).min(rows as i32 - 1) as usize;
                        for kx in -half..=half {
                            let c = (col as i32 + kx).max(0).min(cols as i32 - 1) as usize;

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
        0.3 * ((size as f64 - 1.0) * 0.5 - 1.0) + 0.8
    } else {
        sigma
    };

    let half = size / 2;
    let mut kernel = Vec::with_capacity(size as usize);
    let mut sum = 0.0;

    for i in -half..=half {
        let x = i as f64;
        let value = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(value as f32);
        sum += value;
    }

    // Normalize
    for val in &mut kernel {
        *val /= sum as f32;
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
            for col in 0..cols {
                let mut sums = [0f32; 4];

                for (i, &k) in kernel_x.iter().enumerate() {
                    let offset = i as i32 - half_x as i32;
                    let c = (col as i32 + offset).max(0).min(cols as i32 - 1) as usize;

                    let src_idx = (row * cols + c) * channels;
                    let pixel = &src_data[src_idx..src_idx + channels];

                    match channels {
                        1 => {
                            sums[0] += pixel[0] as f32 * k;
                        }
                        3 => {
                            sums[0] += pixel[0] as f32 * k;
                            sums[1] += pixel[1] as f32 * k;
                            sums[2] += pixel[2] as f32 * k;
                        }
                        4 => {
                            sums[0] += pixel[0] as f32 * k;
                            sums[1] += pixel[1] as f32 * k;
                            sums[2] += pixel[2] as f32 * k;
                            sums[3] += pixel[3] as f32 * k;
                        }
                        _ => {
                            for ch in 0..channels {
                                sums[ch] += pixel[ch] as f32 * k;
                            }
                        }
                    }
                }

                let temp_idx = col * channels;
                let temp_pixel = &mut temp_row[temp_idx..temp_idx + channels];

                match channels {
                    1 => {
                        temp_pixel[0] = sums[0].round().clamp(0.0, 255.0) as u8;
                    }
                    3 => {
                        temp_pixel[0] = sums[0].round().clamp(0.0, 255.0) as u8;
                        temp_pixel[1] = sums[1].round().clamp(0.0, 255.0) as u8;
                        temp_pixel[2] = sums[2].round().clamp(0.0, 255.0) as u8;
                    }
                    4 => {
                        temp_pixel[0] = sums[0].round().clamp(0.0, 255.0) as u8;
                        temp_pixel[1] = sums[1].round().clamp(0.0, 255.0) as u8;
                        temp_pixel[2] = sums[2].round().clamp(0.0, 255.0) as u8;
                        temp_pixel[3] = sums[3].round().clamp(0.0, 255.0) as u8;
                    }
                    _ => {
                        for ch in 0..channels {
                            temp_pixel[ch] = sums[ch].round().clamp(0.0, 255.0) as u8;
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
            for col in 0..cols {
                let mut sums = [0f32; 4];

                for (i, &k) in kernel_y.iter().enumerate() {
                    let offset = i as i32 - half_y as i32;
                    let r = (row as i32 + offset).max(0).min(rows as i32 - 1) as usize;

                    let temp_idx = (r * cols + col) * channels;
                    let pixel = &temp_data[temp_idx..temp_idx + channels];

                    match channels {
                        1 => {
                            sums[0] += pixel[0] as f32 * k;
                        }
                        3 => {
                            sums[0] += pixel[0] as f32 * k;
                            sums[1] += pixel[1] as f32 * k;
                            sums[2] += pixel[2] as f32 * k;
                        }
                        4 => {
                            sums[0] += pixel[0] as f32 * k;
                            sums[1] += pixel[1] as f32 * k;
                            sums[2] += pixel[2] as f32 * k;
                            sums[3] += pixel[3] as f32 * k;
                        }
                        _ => {
                            for ch in 0..channels {
                                sums[ch] += pixel[ch] as f32 * k;
                            }
                        }
                    }
                }

                let dst_idx = col * channels;
                let dst_pixel = &mut dst_row[dst_idx..dst_idx + channels];

                match channels {
                    1 => {
                        dst_pixel[0] = sums[0].round().clamp(0.0, 255.0) as u8;
                    }
                    3 => {
                        dst_pixel[0] = sums[0].round().clamp(0.0, 255.0) as u8;
                        dst_pixel[1] = sums[1].round().clamp(0.0, 255.0) as u8;
                        dst_pixel[2] = sums[2].round().clamp(0.0, 255.0) as u8;
                    }
                    4 => {
                        dst_pixel[0] = sums[0].round().clamp(0.0, 255.0) as u8;
                        dst_pixel[1] = sums[1].round().clamp(0.0, 255.0) as u8;
                        dst_pixel[2] = sums[2].round().clamp(0.0, 255.0) as u8;
                        dst_pixel[3] = sums[3].round().clamp(0.0, 255.0) as u8;
                    }
                    _ => {
                        for ch in 0..channels {
                            dst_pixel[ch] = sums[ch].round().clamp(0.0, 255.0) as u8;
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
