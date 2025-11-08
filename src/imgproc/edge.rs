use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};

/// Calculate Sobel derivatives
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

/// Calculate Laplacian
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

    // Step 3: Calculate gradient magnitude and direction
    let mut magnitude = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;
    let mut direction = vec![vec![0.0f32; src.cols()]; src.rows()];

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let gx = grad_x.at(row, col)?[0] as f32;
            let gy = grad_y.at(row, col)?[0] as f32;

            let mag = ((gx * gx + gy * gy) as f32).sqrt();
            let mag_pixel = magnitude.at_mut(row, col)?;
            mag_pixel[0] = mag.min(255.0) as u8;

            direction[row][col] = gy.atan2(gx);
        }
    }

    // Step 4: Non-maximum suppression
    let mut suppressed = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    for row in 1..src.rows() - 1 {
        for col in 1..src.cols() - 1 {
            let mag = magnitude.at(row, col)?[0];
            let angle = direction[row][col];

            // Quantize angle to 0, 45, 90, 135 degrees
            let angle_deg = (angle * 180.0 / std::f32::consts::PI + 180.0) % 180.0;

            let (n1, n2) = if angle_deg < 22.5 || angle_deg >= 157.5 {
                // 0 degrees - horizontal
                (magnitude.at(row, col - 1)?[0], magnitude.at(row, col + 1)?[0])
            } else if angle_deg < 67.5 {
                // 45 degrees
                (magnitude.at(row - 1, col + 1)?[0], magnitude.at(row + 1, col - 1)?[0])
            } else if angle_deg < 112.5 {
                // 90 degrees - vertical
                (magnitude.at(row - 1, col)?[0], magnitude.at(row + 1, col)?[0])
            } else {
                // 135 degrees
                (magnitude.at(row - 1, col - 1)?[0], magnitude.at(row + 1, col + 1)?[0])
            };

            let sup_pixel = suppressed.at_mut(row, col)?;
            if mag >= n1 && mag >= n2 {
                sup_pixel[0] = mag;
            } else {
                sup_pixel[0] = 0;
            }
        }
    }

    // Step 5: Double threshold and edge tracking by hysteresis
    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    let low_threshold = threshold1;
    let high_threshold = threshold2;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let mag = suppressed.at(row, col)?[0] as f64;
            let dst_pixel = dst.at_mut(row, col)?;

            if mag >= high_threshold {
                dst_pixel[0] = 255; // Strong edge
            } else if mag >= low_threshold {
                // Weak edge - check if connected to strong edge
                let mut connected = false;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        if dy == 0 && dx == 0 {
                            continue;
                        }

                        let ny = row as i32 + dy;
                        let nx = col as i32 + dx;

                        if ny >= 0 && ny < src.rows() as i32 && nx >= 0 && nx < src.cols() as i32 {
                            let neighbor = suppressed.at(ny as usize, nx as usize)?[0] as f64;
                            if neighbor >= high_threshold {
                                connected = true;
                                break;
                            }
                        }
                    }
                    if connected {
                        break;
                    }
                }

                dst_pixel[0] = if connected { 255 } else { 0 };
            } else {
                dst_pixel[0] = 0; // Not an edge
            }
        }
    }

    Ok(())
}

/// Scharr derivative filter (more accurate than Sobel 3x3)
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
