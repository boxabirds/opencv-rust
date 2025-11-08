use crate::core::{Mat, MatDepth};
use crate::core::types::Size;
use crate::error::{Error, Result};

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

/// Apply box blur (simple averaging)
pub fn blur(src: &Mat, dst: &mut Mat, ksize: Size) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "blur only supports U8 depth".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    let half_w = ksize.width / 2;
    let half_h = ksize.height / 2;
    let kernel_size = (ksize.width * ksize.height) as f32;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let mut sums = vec![0f32; src.channels()];

            for ky in -half_h..=half_h {
                for kx in -half_w..=half_w {
                    let r = (row as i32 + ky).max(0).min(src.rows() as i32 - 1) as usize;
                    let c = (col as i32 + kx).max(0).min(src.cols() as i32 - 1) as usize;

                    let pixel = src.at(r, c)?;
                    for ch in 0..src.channels() {
                        sums[ch] += pixel[ch] as f32;
                    }
                }
            }

            let dst_pixel = dst.at_mut(row, col)?;
            for ch in 0..src.channels() {
                dst_pixel[ch] = (sums[ch] / kernel_size) as u8;
            }
        }
    }

    Ok(())
}

/// Apply median blur
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

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    let half = ksize / 2;
    let kernel_area = (ksize * ksize) as usize;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            for ch in 0..src.channels() {
                let mut values = Vec::with_capacity(kernel_area);

                for ky in -half..=half {
                    for kx in -half..=half {
                        let r = (row as i32 + ky).max(0).min(src.rows() as i32 - 1) as usize;
                        let c = (col as i32 + kx).max(0).min(src.cols() as i32 - 1) as usize;

                        let pixel = src.at(r, c)?;
                        values.push(pixel[ch]);
                    }
                }

                values.sort_unstable();
                let median = values[kernel_area / 2];

                let dst_pixel = dst.at_mut(row, col)?;
                dst_pixel[ch] = median;
            }
        }
    }

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

/// Apply separable filter (for efficiency)
fn apply_separable_filter(
    src: &Mat,
    dst: &mut Mat,
    kernel_x: &[f32],
    kernel_y: &[f32],
) -> Result<()> {
    // First apply horizontal kernel
    let temp = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;
    let mut temp = temp;

    let half_x = kernel_x.len() / 2;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let mut sums = vec![0f32; src.channels()];

            for (i, &k) in kernel_x.iter().enumerate() {
                let offset = i as i32 - half_x as i32;
                let c = (col as i32 + offset).max(0).min(src.cols() as i32 - 1) as usize;

                let pixel = src.at(row, c)?;
                for ch in 0..src.channels() {
                    sums[ch] += pixel[ch] as f32 * k;
                }
            }

            let temp_pixel = temp.at_mut(row, col)?;
            for ch in 0..src.channels() {
                temp_pixel[ch] = sums[ch].round().min(255.0).max(0.0) as u8;
            }
        }
    }

    // Then apply vertical kernel
    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    let half_y = kernel_y.len() / 2;

    for row in 0..temp.rows() {
        for col in 0..temp.cols() {
            let mut sums = vec![0f32; temp.channels()];

            for (i, &k) in kernel_y.iter().enumerate() {
                let offset = i as i32 - half_y as i32;
                let r = (row as i32 + offset).max(0).min(temp.rows() as i32 - 1) as usize;

                let pixel = temp.at(r, col)?;
                for ch in 0..temp.channels() {
                    sums[ch] += pixel[ch] as f32 * k;
                }
            }

            let dst_pixel = dst.at_mut(row, col)?;
            for ch in 0..temp.channels() {
                dst_pixel[ch] = sums[ch].round().min(255.0).max(0.0) as u8;
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
