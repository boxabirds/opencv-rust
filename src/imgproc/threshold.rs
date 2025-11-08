use crate::core::{Mat, MatDepth};
use crate::core::types::ThresholdType;
use crate::error::{Error, Result};

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

    let thresh = thresh as u8;
    let maxval = maxval as u8;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let src_pixel = src.at(row, col)?;
            let dst_pixel = dst.at_mut(row, col)?;

            for ch in 0..src.channels() {
                let value = src_pixel[ch];

                dst_pixel[ch] = match thresh_type {
                    ThresholdType::Binary => {
                        if value > thresh {
                            maxval
                        } else {
                            0
                        }
                    }
                    ThresholdType::BinaryInv => {
                        if value > thresh {
                            0
                        } else {
                            maxval
                        }
                    }
                    ThresholdType::Trunc => {
                        if value > thresh {
                            thresh
                        } else {
                            value
                        }
                    }
                    ThresholdType::ToZero => {
                        if value > thresh {
                            value
                        } else {
                            0
                        }
                    }
                    ThresholdType::ToZeroInv => {
                        if value > thresh {
                            0
                        } else {
                            value
                        }
                    }
                };
            }
        }
    }

    Ok(thresh as f64)
}

/// Apply adaptive threshold
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

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    let half = block_size / 2;
    let maxval = maxval as u8;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let mut sum = 0f64;
            let mut count = 0;

            for ky in -half..=half {
                for kx in -half..=half {
                    let r = (row as i32 + ky).max(0).min(src.rows() as i32 - 1) as usize;
                    let c = (col as i32 + kx).max(0).min(src.cols() as i32 - 1) as usize;

                    let pixel = src.at(r, c)?;
                    sum += pixel[0] as f64;
                    count += 1;
                }
            }

            let local_thresh = match method {
                AdaptiveThresholdMethod::Mean => (sum / count as f64) - c,
                AdaptiveThresholdMethod::Gaussian => {
                    // Simplified - would normally use weighted Gaussian
                    (sum / count as f64) - c
                }
            };

            let src_pixel = src.at(row, col)?;
            let value = src_pixel[0];

            let dst_pixel = dst.at_mut(row, col)?;
            dst_pixel[0] = match thresh_type {
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
                _ => {
                    return Err(Error::UnsupportedOperation(
                        "adaptive_threshold only supports Binary and BinaryInv types".to_string(),
                    ));
                }
            };
        }
    }

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
