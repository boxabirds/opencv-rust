use crate::core::{Mat, MatDepth};
use crate::core::types::ThresholdType;
use crate::error::{Error, Result};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Apply threshold to an image
pub fn threshold(
    src: &Mat,
    dst: &mut Mat,
    thresh: f64,
    maxval: f64,
    thresh_type: ThresholdType,
) -> Result<f64> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "threshold only supports U8 depth".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    // Clamp and convert f64 threshold values to u8 range
    let thresh_u8 = thresh.clamp(0.0, 255.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let thresh_u8 = thresh_u8 as u8;

    let maxval_u8 = maxval.clamp(0.0, 255.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let maxval_u8 = maxval_u8 as u8;
    let rows = src.rows();
    let cols = src.cols();
    let channels = src.channels();

    // Parallel processing for performance
    rayon::scope(|_s| {
        let src_data = src.data();
        let dst_data = dst.data_mut();
        let row_size = cols * channels;

        dst_data.par_chunks_mut(row_size).enumerate().for_each(|(row, dst_row_data)| {
            for col in 0..cols {
                let src_idx = (row * cols + col) * channels;
                let dst_idx = col * channels;

                for ch in 0..channels {
                    let value = src_data[src_idx + ch];

                    dst_row_data[dst_idx + ch] = match thresh_type {
                        ThresholdType::Binary => {
                            if value > thresh_u8 { maxval_u8 } else { 0 }
                        }
                        ThresholdType::BinaryInv => {
                            if value > thresh_u8 { 0 } else { maxval_u8 }
                        }
                        ThresholdType::Trunc => {
                            if value > thresh_u8 { thresh_u8 } else { value }
                        }
                        ThresholdType::ToZero => {
                            if value > thresh_u8 { value } else { 0 }
                        }
                        ThresholdType::ToZeroInv => {
                            if value > thresh_u8 { 0 } else { value }
                        }
                    };
                }
            }
        });
    });

    Ok(thresh)
}

/// Apply adaptive threshold - optimized with rayon parallelization
/// Adaptive threshold with GPU acceleration (async for WASM)
pub async fn adaptive_threshold_async(
    src: &Mat,
    dst: &mut Mat,
    maxval: f64,
    method: AdaptiveThresholdMethod,
    thresh_type: ThresholdType,
    block_size: i32,
    c: f64,
    use_gpu: bool,
) -> Result<()> {
    // Try GPU if requested, available, and method is Mean
    if use_gpu && method == AdaptiveThresholdMethod::Mean {
        #[cfg(feature = "gpu")]
        {
            use crate::gpu::ops::adaptive_threshold_gpu_async;
            // Clamp and convert parameters for GPU call
            let maxval_u8 = maxval.clamp(0.0, 255.0);
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let maxval_u8 = maxval_u8 as u8;

            let c_i32 = c.clamp(i32::MIN as f64, i32::MAX as f64);
            #[allow(clippy::cast_possible_truncation)]
            let c_i32 = c_i32 as i32;

            match adaptive_threshold_gpu_async(src, dst, maxval_u8, block_size, c_i32).await {
                Ok(()) => return Ok(()),
                Err(_) => {
                    // Fall through to CPU
                }
            }
        }
    }

    // CPU fallback
    adaptive_threshold(src, dst, maxval, method, thresh_type, block_size, c)
}

/// Adaptive threshold (CPU-only, sync)
pub fn adaptive_threshold(
    src: &Mat,
    dst: &mut Mat,
    maxval: f64,
    method: AdaptiveThresholdMethod,
    thresh_type: ThresholdType,
    block_size: i32,
    c: f64,
) -> Result<()> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "adaptive_threshold only works on grayscale images".to_string(),
        ));
    }

    if block_size % 2 == 0 || block_size < 3 {
        return Err(Error::InvalidParameter(
            "block_size must be odd and >= 3".to_string(),
        ));
    }

    // Validate threshold type
    match thresh_type {
        ThresholdType::Binary | ThresholdType::BinaryInv => {}
        _ => {
            return Err(Error::UnsupportedOperation(
                "adaptive_threshold only supports Binary and BinaryInv types".to_string(),
            ));
        }
    }

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    let rows = src.rows();
    let cols = src.cols();
    let half = block_size / 2;
    // Clamp and convert maxval to u8 range
    let maxval_clamped = maxval.clamp(0.0, 255.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let maxval = maxval_clamped as u8;

    // Use rayon::scope to safely share references
    rayon::scope(|_s| {
        let dst_data = dst.data_mut();
        let src_data = src.data();

        dst_data.par_chunks_mut(cols).enumerate().for_each(|(row, dst_row)| {
            for (col, dst_pixel) in dst_row.iter_mut().enumerate() {
                let mut sum = 0u32;
                let mut count = 0u32;

                // Calculate local threshold from neighborhood
                for ky in -half..=half {
                    // Convert for neighbor offset arithmetic
                    let row_i32 = i32::try_from(row).unwrap_or(i32::MAX);
                    let rows_i32 = i32::try_from(rows).unwrap_or(i32::MAX);
                    let r_i32 = (row_i32 + ky).max(0).min(rows_i32 - 1);
                    let r = usize::try_from(r_i32).unwrap_or(0);

                    for kx in -half..=half {
                        let col_i32 = i32::try_from(col).unwrap_or(i32::MAX);
                        let cols_i32 = i32::try_from(cols).unwrap_or(i32::MAX);
                        let c_i32 = (col_i32 + kx).max(0).min(cols_i32 - 1);
                        let c_offset = usize::try_from(c_i32).unwrap_or(0);

                        let src_idx = r * cols + c_offset;
                        sum += u32::from(src_data[src_idx]);
                        count += 1;
                    }
                }

                let local_thresh = match method {
                    AdaptiveThresholdMethod::Mean => (f64::from(sum) / f64::from(count)) - c,
                    AdaptiveThresholdMethod::Gaussian => {
                        // Simplified - use mean for now (would normally use weighted Gaussian)
                        (f64::from(sum) / f64::from(count)) - c
                    }
                };

                let src_idx = row * cols + col;
                let value = src_data[src_idx];

                *dst_pixel = match thresh_type {
                    ThresholdType::Binary => {
                        if f64::from(value) > local_thresh {
                            maxval
                        } else {
                            0
                        }
                    }
                    ThresholdType::BinaryInv => {
                        if f64::from(value) > local_thresh {
                            0
                        } else {
                            maxval
                        }
                    }
                    _ => 0, // Already validated above
                };
            }
        });
    });

    Ok(())
}

/// Adaptive threshold methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveThresholdMethod {
    Mean,
    Gaussian,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_threshold_binary() {
        let src = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        threshold(&src, &mut dst, 100.0, 255.0, ThresholdType::Binary).unwrap();

        let pixel = dst.at(50, 50).unwrap();
        assert_eq!(pixel[0], 255);
    }

    #[test]
    fn test_adaptive_threshold() {
        let src = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        adaptive_threshold(
            &src,
            &mut dst,
            255.0,
            AdaptiveThresholdMethod::Mean,
            ThresholdType::Binary,
            5,
            2.0,
        )
        .unwrap();

        assert_eq!(dst.rows(), src.rows());
        assert_eq!(dst.cols(), src.cols());
    }
}
