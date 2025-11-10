use crate::core::{Mat, MatDepth};
use crate::core::types::Size;
use crate::error::{Error, Result};

#[cfg(feature = "gpu")]
use crate::gpu::ops::{erode_gpu_async, dilate_gpu_async};

/// Morphological operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphType {
    Erode,
    Dilate,
    Open,
    Close,
    Gradient,
    TopHat,
    BlackHat,
}

/// Morphological structuring element shapes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphShape {
    Rect,
    Cross,
    Ellipse,
}

/// Get structuring element for morphological operations
pub fn get_structuring_element(shape: MorphShape, ksize: Size) -> Vec<Vec<bool>> {
    let rows = ksize.height as usize;
    let cols = ksize.width as usize;
    let mut kernel = vec![vec![false; cols]; rows];

    let center_y = rows / 2;
    let center_x = cols / 2;

    match shape {
        MorphShape::Rect => {
            for row in &mut kernel {
                for elem in row {
                    *elem = true;
                }
            }
        }
        MorphShape::Cross => {
            for (y, row) in kernel.iter_mut().enumerate() {
                for (x, elem) in row.iter_mut().enumerate() {
                    *elem = y == center_y || x == center_x;
                }
            }
        }
        MorphShape::Ellipse => {
            let a = center_x as f64;
            let b = center_y as f64;

            for (y, row) in kernel.iter_mut().enumerate() {
                for (x, elem) in row.iter_mut().enumerate() {
                    let dx = (x as f64 - center_x as f64) / a;
                    let dy = (y as f64 - center_y as f64) / b;
                    *elem = dx * dx + dy * dy <= 1.0;
                }
            }
        }
    }

    kernel
}

/// Erode image using structuring element
pub fn erode(src: &Mat, dst: &mut Mat, kernel: &[Vec<bool>]) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "erode only supports U8 depth".to_string(),
        ));
    }

    if kernel.is_empty() || kernel[0].is_empty() {
        return Err(Error::InvalidParameter("Kernel is empty".to_string()));
    }

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    let k_height = kernel.len();
    let k_width = kernel[0].len();
    let half_h = k_height / 2;
    let half_w = k_width / 2;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let mut min_vals = vec![255u8; src.channels()];

            for ky in 0..k_height {
                for kx in 0..k_width {
                    if !kernel[ky][kx] {
                        continue;
                    }

                    let y = row as i32 + ky as i32 - half_h as i32;
                    let x = col as i32 + kx as i32 - half_w as i32;

                    if y >= 0 && y < src.rows() as i32 && x >= 0 && x < src.cols() as i32 {
                        let pixel = src.at(y as usize, x as usize)?;
                        for ch in 0..src.channels() {
                            min_vals[ch] = min_vals[ch].min(pixel[ch]);
                        }
                    }
                }
            }

            let dst_pixel = dst.at_mut(row, col)?;
            for (ch, &val) in min_vals.iter().enumerate() {
                dst_pixel[ch] = val;
            }
        }
    }

    Ok(())
}

/// Dilate image using structuring element
pub fn dilate(src: &Mat, dst: &mut Mat, kernel: &[Vec<bool>]) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "dilate only supports U8 depth".to_string(),
        ));
    }

    if kernel.is_empty() || kernel[0].is_empty() {
        return Err(Error::InvalidParameter("Kernel is empty".to_string()));
    }

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    let k_height = kernel.len();
    let k_width = kernel[0].len();
    let half_h = k_height / 2;
    let half_w = k_width / 2;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let mut max_vals = vec![0u8; src.channels()];

            for ky in 0..k_height {
                for kx in 0..k_width {
                    if !kernel[ky][kx] {
                        continue;
                    }

                    let y = row as i32 + ky as i32 - half_h as i32;
                    let x = col as i32 + kx as i32 - half_w as i32;

                    if y >= 0 && y < src.rows() as i32 && x >= 0 && x < src.cols() as i32 {
                        let pixel = src.at(y as usize, x as usize)?;
                        for ch in 0..src.channels() {
                            max_vals[ch] = max_vals[ch].max(pixel[ch]);
                        }
                    }
                }
            }

            let dst_pixel = dst.at_mut(row, col)?;
            for (ch, &val) in max_vals.iter().enumerate() {
                dst_pixel[ch] = val;
            }
        }
    }

    Ok(())
}

/// Perform advanced morphological operations
pub fn morphology_ex(
    src: &Mat,
    dst: &mut Mat,
    op: MorphType,
    kernel: &[Vec<bool>],
) -> Result<()> {
    match op {
        MorphType::Erode => erode(src, dst, kernel),
        MorphType::Dilate => dilate(src, dst, kernel),
        MorphType::Open => {
            // Opening: erosion followed by dilation
            let mut temp = Mat::new(1, 1, 1, MatDepth::U8)?;
            erode(src, &mut temp, kernel)?;
            dilate(&temp, dst, kernel)
        }
        MorphType::Close => {
            // Closing: dilation followed by erosion
            let mut temp = Mat::new(1, 1, 1, MatDepth::U8)?;
            dilate(src, &mut temp, kernel)?;
            erode(&temp, dst, kernel)
        }
        MorphType::Gradient => {
            // Morphological gradient: dilation - erosion
            let mut dilated = Mat::new(1, 1, 1, MatDepth::U8)?;
            let mut eroded = Mat::new(1, 1, 1, MatDepth::U8)?;
            dilate(src, &mut dilated, kernel)?;
            erode(src, &mut eroded, kernel)?;

            *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

            for row in 0..src.rows() {
                for col in 0..src.cols() {
                    let d_pix = dilated.at(row, col)?;
                    let e_pix = eroded.at(row, col)?;
                    let dst_pix = dst.at_mut(row, col)?;

                    for ch in 0..src.channels() {
                        dst_pix[ch] = d_pix[ch].saturating_sub(e_pix[ch]);
                    }
                }
            }
            Ok(())
        }
        MorphType::TopHat => {
            // Top hat: source - opening
            let mut opened = Mat::new(1, 1, 1, MatDepth::U8)?;
            morphology_ex(src, &mut opened, MorphType::Open, kernel)?;

            *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

            for row in 0..src.rows() {
                for col in 0..src.cols() {
                    let s_pix = src.at(row, col)?;
                    let o_pix = opened.at(row, col)?;
                    let dst_pix = dst.at_mut(row, col)?;

                    for ch in 0..src.channels() {
                        dst_pix[ch] = s_pix[ch].saturating_sub(o_pix[ch]);
                    }
                }
            }
            Ok(())
        }
        MorphType::BlackHat => {
            // Black hat: closing - source
            let mut closed = Mat::new(1, 1, 1, MatDepth::U8)?;
            morphology_ex(src, &mut closed, MorphType::Close, kernel)?;

            *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

            for row in 0..src.rows() {
                for col in 0..src.cols() {
                    let c_pix = closed.at(row, col)?;
                    let s_pix = src.at(row, col)?;
                    let dst_pix = dst.at_mut(row, col)?;

                    for ch in 0..src.channels() {
                        dst_pix[ch] = c_pix[ch].saturating_sub(s_pix[ch]);
                    }
                }
            }
            Ok(())
        }
    }
}

/// Async version of erode with optional GPU acceleration
/// For simple rectangular kernels, uses GPU if available and use_gpu is true
pub async fn erode_async(
    src: &Mat,
    dst: &mut Mat,
    kernel: &[Vec<bool>],
    use_gpu: bool,
) -> Result<()> {
    // Try GPU if requested, available, and kernel is simple (rectangular)
    if use_gpu {
        #[cfg(feature = "gpu")]
        {
            // Check if kernel is rectangular (all true)
            let is_rect = kernel.iter().all(|row| row.iter().all(|&v| v));
            if is_rect && !kernel.is_empty() && !kernel[0].is_empty() {
                let ksize = kernel.len() as i32; // Assume square kernel
                match erode_gpu_async(src, dst, ksize).await {
                    Ok(()) => return Ok(()),
                    Err(_) => {
                        // Fall through to CPU
                    }
                }
            }
        }
    }

    // CPU fallback
    erode(src, dst, kernel)
}

/// Async version of dilate with optional GPU acceleration
/// For simple rectangular kernels, uses GPU if available and use_gpu is true
pub async fn dilate_async(
    src: &Mat,
    dst: &mut Mat,
    kernel: &[Vec<bool>],
    use_gpu: bool,
) -> Result<()> {
    // Try GPU if requested, available, and kernel is simple (rectangular)
    if use_gpu {
        #[cfg(feature = "gpu")]
        {
            // Check if kernel is rectangular (all true)
            let is_rect = kernel.iter().all(|row| row.iter().all(|&v| v));
            if is_rect && !kernel.is_empty() && !kernel[0].is_empty() {
                let ksize = kernel.len() as i32; // Assume square kernel
                match dilate_gpu_async(src, dst, ksize).await {
                    Ok(()) => return Ok(()),
                    Err(_) => {
                        // Fall through to CPU
                    }
                }
            }
        }
    }

    // CPU fallback
    dilate(src, dst, kernel)
}

/// Async version of morphology_ex with optional GPU acceleration
pub async fn morphology_ex_async(
    src: &Mat,
    dst: &mut Mat,
    op: MorphType,
    kernel: &[Vec<bool>],
    use_gpu: bool,
) -> Result<()> {
    match op {
        MorphType::Erode => erode_async(src, dst, kernel, use_gpu).await,
        MorphType::Dilate => dilate_async(src, dst, kernel, use_gpu).await,
        MorphType::Open => {
            // Opening: erosion followed by dilation
            let mut temp = Mat::new(1, 1, 1, MatDepth::U8)?;
            erode_async(src, &mut temp, kernel, use_gpu).await?;
            dilate_async(&temp, dst, kernel, use_gpu).await
        }
        MorphType::Close => {
            // Closing: dilation followed by erosion
            let mut temp = Mat::new(1, 1, 1, MatDepth::U8)?;
            dilate_async(src, &mut temp, kernel, use_gpu).await?;
            erode_async(&temp, dst, kernel, use_gpu).await
        }
        MorphType::Gradient => {
            // Morphological gradient: dilation - erosion
            let mut dilated = Mat::new(1, 1, 1, MatDepth::U8)?;
            let mut eroded = Mat::new(1, 1, 1, MatDepth::U8)?;
            dilate_async(src, &mut dilated, kernel, use_gpu).await?;
            erode_async(src, &mut eroded, kernel, use_gpu).await?;

            *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

            for row in 0..src.rows() {
                for col in 0..src.cols() {
                    let d_pix = dilated.at(row, col)?;
                    let e_pix = eroded.at(row, col)?;
                    let dst_pix = dst.at_mut(row, col)?;

                    for ch in 0..src.channels() {
                        dst_pix[ch] = d_pix[ch].saturating_sub(e_pix[ch]);
                    }
                }
            }
            Ok(())
        }
        MorphType::TopHat => {
            // Top hat: source - opening
            let mut opened = Mat::new(1, 1, 1, MatDepth::U8)?;
            Box::pin(morphology_ex_async(src, &mut opened, MorphType::Open, kernel, use_gpu)).await?;

            *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

            for row in 0..src.rows() {
                for col in 0..src.cols() {
                    let s_pix = src.at(row, col)?;
                    let o_pix = opened.at(row, col)?;
                    let dst_pix = dst.at_mut(row, col)?;

                    for ch in 0..src.channels() {
                        dst_pix[ch] = s_pix[ch].saturating_sub(o_pix[ch]);
                    }
                }
            }
            Ok(())
        }
        MorphType::BlackHat => {
            // Black hat: closing - source
            let mut closed = Mat::new(1, 1, 1, MatDepth::U8)?;
            Box::pin(morphology_ex_async(src, &mut closed, MorphType::Close, kernel, use_gpu)).await?;

            *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

            for row in 0..src.rows() {
                for col in 0..src.cols() {
                    let c_pix = closed.at(row, col)?;
                    let s_pix = src.at(row, col)?;
                    let dst_pix = dst.at_mut(row, col)?;

                    for ch in 0..src.channels() {
                        dst_pix[ch] = c_pix[ch].saturating_sub(s_pix[ch]);
                    }
                }
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_structuring_element() {
        let kernel = get_structuring_element(MorphShape::Rect, Size::new(3, 3));
        assert_eq!(kernel.len(), 3);
        assert_eq!(kernel[0].len(), 3);
    }

    #[test]
    fn test_erode_dilate() {
        let src = Mat::new_with_default(10, 10, 1, MatDepth::U8, Scalar::all(255.0)).unwrap();
        let kernel = get_structuring_element(MorphShape::Rect, Size::new(3, 3));

        let mut eroded = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
        erode(&src, &mut eroded, &kernel).unwrap();

        let mut dilated = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
        dilate(&src, &mut dilated, &kernel).unwrap();

        assert_eq!(eroded.rows(), src.rows());
        assert_eq!(dilated.rows(), src.rows());
    }

    #[test]
    fn test_morphology_ex() {
        let src = Mat::new_with_default(10, 10, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let kernel = get_structuring_element(MorphShape::Rect, Size::new(3, 3));

        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
        morphology_ex(&src, &mut dst, MorphType::Gradient, &kernel).unwrap();

        assert_eq!(dst.rows(), src.rows());
    }
}
