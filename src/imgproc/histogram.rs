use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};

/// Calculate histogram for single-channel image
pub fn calc_hist(
    image: &Mat,
    hist_size: usize,
    ranges: (f32, f32),
) -> Result<Vec<f32>> {
    if image.channels() != 1 {
        return Err(Error::InvalidParameter(
            "calc_hist only works on single-channel images".to_string(),
        ));
    }

    if image.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "calc_hist only supports U8 depth".to_string(),
        ));
    }

    let mut hist = vec![0.0f32; hist_size];
    let (min_val, max_val) = ranges;
    let range = max_val - min_val;
    // Convert hist_size to f32 for division - acceptable precision loss for reasonable hist sizes
    #[allow(clippy::cast_precision_loss)]
    let bin_width = range / hist_size as f32;

    for row in 0..image.rows() {
        for col in 0..image.cols() {
            let pixel = image.at(row, col)?;
            let val = f32::from(pixel[0]);

            if val >= min_val && val < max_val {
                // Convert f32 bin index to usize - value is non-negative after conditional check
                let bin_f32 = (val - min_val) / bin_width;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let bin = bin_f32 as usize;
                let bin = bin.min(hist_size - 1);
                hist[bin] += 1.0;
            }
        }
    }

    Ok(hist)
}

/// Normalize histogram
pub fn normalize_hist(hist: &mut [f32], alpha: f32, beta: f32) {
    let min_val = hist.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = hist.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    if max_val - min_val > 0.0 {
        for val in hist.iter_mut() {
            *val = alpha + (beta - alpha) * (*val - min_val) / (max_val - min_val);
        }
    }
}

/// Equalize histogram to improve contrast
pub fn equalize_hist(src: &Mat, dst: &mut Mat) -> Result<()> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "equalize_hist only works on single-channel images".to_string(),
        ));
    }

    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "equalize_hist only supports U8 depth".to_string(),
        ));
    }

    // Calculate histogram
    let mut hist = vec![0u32; 256];

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let pixel = src.at(row, col)?;
            hist[pixel[0] as usize] += 1;
        }
    }

    // Calculate cumulative distribution function (CDF)
    let mut cdf = vec![0u32; 256];
    cdf[0] = hist[0];

    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    // Find minimum non-zero CDF value
    let cdf_min = *cdf.iter().find(|&&x| x > 0).unwrap_or(&0);
    // Convert total pixels to u32, saturating if it exceeds u32::MAX
    let total_pixels = u32::try_from(src.rows() * src.cols()).unwrap_or(u32::MAX);

    // Calculate lookup table for equalization
    let mut lut = [0u8; 256];

    for i in 0..256 {
        if total_pixels > cdf_min {
            let normalized = (f64::from((cdf[i].saturating_sub(cdf_min))) / f64::from(total_pixels - cdf_min)) * 255.0;
            // Safe cast: normalized is in [0, 255] range
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let lut_val = normalized as u8;
            lut[i] = lut_val;
        } else {
            lut[i] = 0;
        }
    }

    // Apply equalization
    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let src_pixel = src.at(row, col)?;
            let dst_pixel = dst.at_mut(row, col)?;
            dst_pixel[0] = lut[src_pixel[0] as usize];
        }
    }

    Ok(())
}

/// Compare two histograms using different methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistCompMethod {
    Correlation,
    ChiSquare,
    Intersection,
    Bhattacharyya,
}

pub fn compare_hist(h1: &[f32], h2: &[f32], method: HistCompMethod) -> Result<f64> {
    if h1.len() != h2.len() {
        return Err(Error::InvalidParameter(
            "Histograms must have same size".to_string(),
        ));
    }

    let result = match method {
        HistCompMethod::Correlation => {
            // Pearson correlation coefficient
            // Acceptable precision loss for histogram length conversions
            #[allow(clippy::cast_precision_loss)]
            let len_f32 = h1.len() as f32;
            let mean1: f32 = h1.iter().sum::<f32>() / len_f32;
            let mean2: f32 = h2.iter().sum::<f32>() / len_f32;

            let mut num = 0.0;
            let mut den1 = 0.0;
            let mut den2 = 0.0;

            for i in 0..h1.len() {
                let d1 = h1[i] - mean1;
                let d2 = h2[i] - mean2;

                num += d1 * d2;
                den1 += d1 * d1;
                den2 += d2 * d2;
            }

            if den1 * den2 > 0.0 {
                f64::from(num / (den1 * den2).sqrt())
            } else {
                0.0
            }
        }
        HistCompMethod::ChiSquare => {
            let mut sum = 0.0;

            for i in 0..h1.len() {
                if h1[i] + h2[i] > 0.0 {
                    let diff = h1[i] - h2[i];
                    sum += (diff * diff) / (h1[i] + h2[i]);
                }
            }

            f64::from(sum)
        }
        HistCompMethod::Intersection => {
            let mut sum = 0.0;

            for i in 0..h1.len() {
                sum += h1[i].min(h2[i]);
            }

            f64::from(sum)
        }
        HistCompMethod::Bhattacharyya => {
            let sum1: f32 = h1.iter().sum();
            let sum2: f32 = h2.iter().sum();

            let mut bc = 0.0;

            for i in 0..h1.len() {
                if sum1 > 0.0 && sum2 > 0.0 {
                    bc += ((h1[i] / sum1) * (h2[i] / sum2)).sqrt();
                }
            }

            f64::from((-bc.ln()).max(0.0))
        }
    };

    Ok(result)
}

/// Calculate back projection
pub fn calc_back_project(
    image: &Mat,
    hist: &[f32],
    ranges: (f32, f32),
    dst: &mut Mat,
) -> Result<()> {
    if image.channels() != 1 {
        return Err(Error::InvalidParameter(
            "calc_back_project only works on single-channel images".to_string(),
        ));
    }

    *dst = Mat::new(image.rows(), image.cols(), 1, MatDepth::U8)?;

    let (min_val, max_val) = ranges;
    let range = max_val - min_val;
    // Acceptable precision loss for histogram length conversion
    #[allow(clippy::cast_precision_loss)]
    let bin_width = range / hist.len() as f32;

    // Find max histogram value for normalization
    let max_hist = hist.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    for row in 0..image.rows() {
        for col in 0..image.cols() {
            let pixel = image.at(row, col)?;
            let val = f32::from(pixel[0]);

            if val >= min_val && val < max_val {
                // Convert f32 bin index to usize - non-negative after check
                let bin_f32 = (val - min_val) / bin_width;
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let bin = bin_f32 as usize;
                let bin = bin.min(hist.len() - 1);

                let back_proj_val = if max_hist > 0.0 {
                    let normalized = (hist[bin] / max_hist) * 255.0;
                    // Safe cast: normalized is in [0, 255] range
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let val = normalized as u8;
                    val
                } else {
                    0
                };

                let dst_pixel = dst.at_mut(row, col)?;
                dst_pixel[0] = back_proj_val;
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
    fn test_calc_hist() {
        let img = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let hist = calc_hist(&img, 256, (0.0, 256.0)).unwrap();

        assert_eq!(hist.len(), 256);
        assert!(hist[128] > 0.0);
    }

    #[test]
    fn test_equalize_hist() {
        let src = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(100.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        equalize_hist(&src, &mut dst).unwrap();
        assert_eq!(dst.rows(), src.rows());
    }

    #[test]
    fn test_compare_hist() {
        let h1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let h2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let corr = compare_hist(&h1, &h2, HistCompMethod::Correlation).unwrap();
        assert!((corr - 1.0).abs() < 0.01); // Should be perfectly correlated
    }
}
