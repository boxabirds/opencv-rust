use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Calculate Sobel derivatives with GPU acceleration (async for WASM)
pub async fn sobel_async(
    src: &Mat,
    dst: &mut Mat,
    dx: i32,
    dy: i32,
    ksize: i32,
    use_gpu: bool,
) -> Result<()> {
    // Try GPU if requested and available
    if use_gpu && ksize == 3 {
        #[cfg(feature = "gpu")]
        {
            use crate::gpu::ops::sobel_gpu_async;
            match sobel_gpu_async(src, dst, dx, dy).await {
                Ok(()) => return Ok(()),
                Err(_) => {
                    // Fall through to CPU
                }
            }
        }
    }

    // CPU fallback
    sobel(src, dst, dx, dy, ksize)
}

/// Calculate Sobel derivatives (CPU-only, sync)
pub fn sobel(
    src: &Mat,
    dst: &mut Mat,
    dx: i32,
    dy: i32,
    ksize: i32,
) -> Result<()> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Sobel only works on single-channel images".to_string(),
        ));
    }

    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "Sobel only supports U8 depth".to_string(),
        ));
    }

    if ksize != 3 && ksize != 5 {
        return Err(Error::InvalidParameter(
            "Only kernel sizes 3 and 5 are supported".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    // For simplicity, implementing 3x3 Sobel
    let kernel_x = [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ];

    let kernel_y = [
        [-1.0, -2.0, -1.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    ];

    let rows = src.rows();
    let cols = src.cols();
    let src_data = src.data();

    // Parallel row processing
    rayon::scope(|_s| {
        let dst_data = dst.data_mut();

        dst_data[cols..(rows-1)*cols].par_chunks_mut(cols).enumerate().for_each(|(idx, dst_row)| {
            let row = idx + 1; // Offset by 1 since we skip first row

            for col in 1..cols - 1 {
                let mut sum = 0.0;

                if dx > 0 {
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let y = row + ky - 1;
                            let x = col + kx - 1;
                            let src_idx = y * cols + x;
                            sum += src_data[src_idx] as f64 * kernel_x[ky][kx];
                        }
                    }
                }

                if dy > 0 {
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let y = row + ky - 1;
                            let x = col + kx - 1;
                            let src_idx = y * cols + x;
                            sum += src_data[src_idx] as f64 * kernel_y[ky][kx];
                        }
                    }
                }

                let val = sum.abs().min(255.0).max(0.0) as u8;
                dst_row[col] = val;
            }
        });
    });

    Ok(())
}

/// Calculate Laplacian with GPU acceleration (async for WASM)
pub async fn laplacian_async(
    src: &Mat,
    dst: &mut Mat,
    ksize: i32,
    use_gpu: bool,
) -> Result<()> {
    // Try GPU if requested and available
    if use_gpu {
        #[cfg(feature = "gpu")]
        {
            use crate::gpu::ops::laplacian_gpu_async;
            match laplacian_gpu_async(src, dst).await {
                Ok(()) => return Ok(()),
                Err(_) => {
                    // Fall through to CPU
                }
            }
        }
    }

    // CPU fallback
    laplacian(src, dst, ksize)
}

/// Calculate Laplacian (CPU-only, sync)
pub fn laplacian(src: &Mat, dst: &mut Mat, ksize: i32) -> Result<()> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Laplacian only works on single-channel images".to_string(),
        ));
    }

    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "Laplacian only supports U8 depth".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    // 3x3 Laplacian kernel
    let kernel = [
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0],
    ];

    for row in 1..src.rows() - 1 {
        for col in 1..src.cols() - 1 {
            let mut sum = 0.0;

            for ky in 0..3 {
                for kx in 0..3 {
                    let y = row + ky - 1;
                    let x = col + kx - 1;
                    let pixel = src.at(y, x)?;
                    sum += pixel[0] as f64 * kernel[ky][kx];
                }
            }

            let val = sum.abs().min(255.0).max(0.0) as u8;
            let dst_pixel = dst.at_mut(row, col)?;
            dst_pixel[0] = val;
        }
    }

    Ok(())
}

/// Canny edge detection
pub fn canny(
    src: &Mat,
    dst: &mut Mat,
    threshold1: f64,
    threshold2: f64,
) -> Result<()> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Canny only works on single-channel images".to_string(),
        ));
    }

    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "Canny only supports U8 depth".to_string(),
        ));
    }

    // Step 1: Apply Gaussian blur to reduce noise
    use crate::imgproc::gaussian_blur;
    use crate::core::types::Size;

    let mut blurred = Mat::new(1, 1, 1, MatDepth::U8)?;
    gaussian_blur(src, &mut blurred, Size::new(5, 5), 1.4)?;

    // Step 2: Calculate gradients using Sobel
    let mut grad_x = Mat::new(1, 1, 1, MatDepth::U8)?;
    let mut grad_y = Mat::new(1, 1, 1, MatDepth::U8)?;
    sobel(&blurred, &mut grad_x, 1, 0, 3)?;
    sobel(&blurred, &mut grad_y, 0, 1, 3)?;

    // Step 3: Calculate gradient magnitude and direction together - parallel
    let mut magnitude = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;
    let mut direction = vec![0.0f32; src.rows() * src.cols()];

    let rows = src.rows();
    let cols = src.cols();
    let grad_x_data = grad_x.data();
    let grad_y_data = grad_y.data();
    let magnitude_data = magnitude.data_mut();

    // Compute both magnitude and direction in parallel (better cache locality)
    rayon::scope(|_s| {
        let direction_slice = &mut direction[..];

        magnitude_data.par_chunks_mut(cols)
            .zip(direction_slice.par_chunks_mut(cols))
            .enumerate()
            .for_each(|(row, (mag_row, dir_row))| {
                for col in 0..cols {
                    let idx = row * cols + col;
                    let gx = grad_x_data[idx] as f32;
                    let gy = grad_y_data[idx] as f32;

                    let mag = (gx * gx + gy * gy).sqrt();
                    mag_row[col] = mag.min(255.0) as u8;
                    dir_row[col] = gy.atan2(gx);
                }
            });
    });

    // Step 4: Non-maximum suppression - parallel
    let mut suppressed = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    rayon::scope(|_s| {
        let suppressed_data = suppressed.data_mut();
        let magnitude_data = magnitude.data();
        let direction_slice = &direction[..];

        suppressed_data[cols..(rows-1)*cols]
            .par_chunks_mut(cols)
            .zip(direction_slice[cols..(rows-1)*cols].par_chunks(cols))
            .enumerate()
            .for_each(|(idx, (sup_row, dir_row))| {
                let row = idx + 1;

                for col in 1..cols - 1 {
                    let mag_idx = row * cols + col;
                    let mag = magnitude_data[mag_idx];
                    let angle = dir_row[col];

                    // Quantize angle to 0, 45, 90, 135 degrees
                    let angle_deg = (angle * 180.0 / std::f32::consts::PI + 180.0) % 180.0;

                    let (n1, n2) = if angle_deg < 22.5 || angle_deg >= 157.5 {
                        // 0 degrees - horizontal
                        (magnitude_data[mag_idx - 1], magnitude_data[mag_idx + 1])
                    } else if angle_deg < 67.5 {
                        // 45 degrees
                        (magnitude_data[mag_idx - cols + 1], magnitude_data[mag_idx + cols - 1])
                    } else if angle_deg < 112.5 {
                        // 90 degrees - vertical
                        (magnitude_data[mag_idx - cols], magnitude_data[mag_idx + cols])
                    } else {
                        // 135 degrees
                        (magnitude_data[mag_idx - cols - 1], magnitude_data[mag_idx + cols + 1])
                    };

                    sup_row[col] = if mag >= n1 && mag >= n2 { mag } else { 0 };
                }
            });
    });

    // Step 5: Double threshold and edge tracking by hysteresis - parallel
    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    let low_threshold = threshold1 as u8;
    let high_threshold = threshold2 as u8;

    rayon::scope(|_s| {
        let dst_data = dst.data_mut();
        let suppressed_data = suppressed.data();

        dst_data.par_chunks_mut(cols).enumerate().for_each(|(row, dst_row)| {
            let base_idx = row * cols;

            for col in 0..cols {
                let mag = suppressed_data[base_idx + col];

                if mag >= high_threshold {
                    dst_row[col] = 255; // Strong edge
                } else if mag >= low_threshold {
                    // Weak edge - check if connected to strong edge in 8-neighborhood
                    // Optimized: Check most likely neighbors first (horizontal/vertical before diagonal)
                    let connected =
                        // Same row (most likely)
                        (col > 0 && suppressed_data[base_idx + col - 1] >= high_threshold) ||
                        (col < cols - 1 && suppressed_data[base_idx + col + 1] >= high_threshold) ||
                        // Top/bottom (next most likely)
                        (row > 0 && suppressed_data[base_idx - cols + col] >= high_threshold) ||
                        (row < rows - 1 && suppressed_data[base_idx + cols + col] >= high_threshold) ||
                        // Diagonals (less likely)
                        (row > 0 && col > 0 && suppressed_data[base_idx - cols + col - 1] >= high_threshold) ||
                        (row > 0 && col < cols - 1 && suppressed_data[base_idx - cols + col + 1] >= high_threshold) ||
                        (row < rows - 1 && col > 0 && suppressed_data[base_idx + cols + col - 1] >= high_threshold) ||
                        (row < rows - 1 && col < cols - 1 && suppressed_data[base_idx + cols + col + 1] >= high_threshold);

                    dst_row[col] = if connected { 255 } else { 0 };
                } else {
                    dst_row[col] = 0; // Not an edge
                }
            }
        });
    });

    Ok(())
}

/// Scharr derivative filter with GPU acceleration (async for WASM)
pub async fn scharr_async(
    src: &Mat,
    dst: &mut Mat,
    dx: i32,
    dy: i32,
    use_gpu: bool,
) -> Result<()> {
    // Try GPU if requested and available
    if use_gpu {
        #[cfg(feature = "gpu")]
        {
            use crate::gpu::ops::scharr_gpu_async;
            match scharr_gpu_async(src, dst, dx, dy).await {
                Ok(()) => return Ok(()),
                Err(_) => {
                    // Fall through to CPU
                }
            }
        }
    }

    // CPU fallback
    scharr(src, dst, dx, dy)
}

/// Scharr derivative filter (more accurate than Sobel 3x3) (CPU-only, sync)
pub fn scharr(src: &Mat, dst: &mut Mat, dx: i32, dy: i32) -> Result<()> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Scharr only works on single-channel images".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    let kernel_x = [
        [-3.0, 0.0, 3.0],
        [-10.0, 0.0, 10.0],
        [-3.0, 0.0, 3.0],
    ];

    let kernel_y = [
        [-3.0, -10.0, -3.0],
        [0.0, 0.0, 0.0],
        [3.0, 10.0, 3.0],
    ];

    for row in 1..src.rows() - 1 {
        for col in 1..src.cols() - 1 {
            let mut sum = 0.0;

            if dx > 0 {
                for ky in 0..3 {
                    for kx in 0..3 {
                        let y = row + ky - 1;
                        let x = col + kx - 1;
                        let pixel = src.at(y, x)?;
                        sum += pixel[0] as f64 * kernel_x[ky][kx];
                    }
                }
            }

            if dy > 0 {
                for ky in 0..3 {
                    for kx in 0..3 {
                        let y = row + ky - 1;
                        let x = col + kx - 1;
                        let pixel = src.at(y, x)?;
                        sum += pixel[0] as f64 * kernel_y[ky][kx];
                    }
                }
            }

            let val = sum.abs().min(255.0).max(0.0) as u8;
            let dst_pixel = dst.at_mut(row, col)?;
            dst_pixel[0] = val;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_sobel() {
        let src = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        sobel(&src, &mut dst, 1, 0, 3).unwrap();
        assert_eq!(dst.rows(), src.rows());
    }

    #[test]
    fn test_laplacian() {
        let src = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        laplacian(&src, &mut dst, 3).unwrap();
        assert_eq!(dst.rows(), src.rows());
    }

    #[test]
    fn test_canny() {
        let src = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        canny(&src, &mut dst, 50.0, 150.0).unwrap();
        assert_eq!(dst.rows(), src.rows());
    }
}
