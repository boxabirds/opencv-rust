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

    let thresh_u8 = thresh as u8;
    let maxval_u8 = maxval as u8;
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
    let maxval = maxval as u8;

    // Use rayon::scope to safely share references
    rayon::scope(|_s| {
        let dst_data = dst.data_mut();
        let src_data = src.data();

        dst_data.par_chunks_mut(cols).enumerate().for_each(|(row, dst_row)| {
            for col in 0..cols {
                let mut sum = 0u32;
                let mut count = 0u32;

                // Calculate local threshold from neighborhood
                for ky in -half..=half {
                    let r = (row as i32 + ky).max(0).min(rows as i32 - 1) as usize;
                    for kx in -half..=half {
                        let c_offset = (col as i32 + kx).max(0).min(cols as i32 - 1) as usize;

                        let src_idx = r * cols + c_offset;
                        sum += src_data[src_idx] as u32;
                        count += 1;
                    }
                }

                let local_thresh = match method {
                    AdaptiveThresholdMethod::Mean => (sum as f64 / count as f64) - c,
                    AdaptiveThresholdMethod::Gaussian => {
                        // Simplified - use mean for now (would normally use weighted Gaussian)
                        (sum as f64 / count as f64) - c
                    }
                };

                let src_idx = row * cols + col;
                let value = src_data[src_idx];

                dst_row[col] = match thresh_type {
                    ThresholdType::Binary => {
                        if value as f64 > local_thresh {
                            maxval
                        } else {
                            0
                        }
                    }
                    ThresholdType::BinaryInv => {
                        if value as f64 > local_thresh {
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
