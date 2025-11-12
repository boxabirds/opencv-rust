use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};
use crate::core::types::Scalar;

/// Add two matrices element-wise
pub fn add(src1: &Mat, src2: &Mat, dst: &mut Mat) -> Result<()> {
    if src1.rows() != src2.rows() || src1.cols() != src2.cols() {
        return Err(Error::InvalidDimensions(
            "Matrices must have same dimensions".to_string(),
        ));
    }

    if src1.channels() != src2.channels() {
        return Err(Error::InvalidParameter(
            "Matrices must have same number of channels".to_string(),
        ));
    }

    if src1.depth() != MatDepth::U8 || src2.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "add only supports U8 depth".to_string(),
        ));
    }

    *dst = Mat::new(src1.rows(), src1.cols(), src1.channels(), src1.depth())?;

    for row in 0..src1.rows() {
        for col in 0..src1.cols() {
            let p1 = src1.at(row, col)?;
            let p2 = src2.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src1.channels() {
                pd[ch] = p1[ch].saturating_add(p2[ch]);
            }
        }
    }

    Ok(())
}

/// Subtract two matrices element-wise
pub fn subtract(src1: &Mat, src2: &Mat, dst: &mut Mat) -> Result<()> {
    if src1.rows() != src2.rows() || src1.cols() != src2.cols() {
        return Err(Error::InvalidDimensions(
            "Matrices must have same dimensions".to_string(),
        ));
    }

    if src1.depth() != MatDepth::U8 || src2.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "subtract only supports U8 depth".to_string(),
        ));
    }

    *dst = Mat::new(src1.rows(), src1.cols(), src1.channels(), src1.depth())?;

    for row in 0..src1.rows() {
        for col in 0..src1.cols() {
            let p1 = src1.at(row, col)?;
            let p2 = src2.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src1.channels() {
                pd[ch] = p1[ch].saturating_sub(p2[ch]);
            }
        }
    }

    Ok(())
}

/// Multiply two matrices element-wise
pub fn multiply(src1: &Mat, src2: &Mat, dst: &mut Mat, scale: f64) -> Result<()> {
    if src1.rows() != src2.rows() || src1.cols() != src2.cols() {
        return Err(Error::InvalidDimensions(
            "Matrices must have same dimensions".to_string(),
        ));
    }

    *dst = Mat::new(src1.rows(), src1.cols(), src1.channels(), src1.depth())?;

    for row in 0..src1.rows() {
        for col in 0..src1.cols() {
            let p1 = src1.at(row, col)?;
            let p2 = src2.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src1.channels() {
                let val = (f64::from(p1[ch]) * f64::from(p2[ch]) * scale).clamp(0.0, 255.0);
                // Safe cast: value is clamped to valid u8 range
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let byte_val = val as u8;
                pd[ch] = byte_val;
            }
        }
    }

    Ok(())
}

/// Bitwise AND operation
pub fn bitwise_and(src1: &Mat, src2: &Mat, dst: &mut Mat) -> Result<()> {
    if src1.rows() != src2.rows() || src1.cols() != src2.cols() {
        return Err(Error::InvalidDimensions(
            "Matrices must have same dimensions".to_string(),
        ));
    }

    *dst = Mat::new(src1.rows(), src1.cols(), src1.channels(), src1.depth())?;

    for row in 0..src1.rows() {
        for col in 0..src1.cols() {
            let p1 = src1.at(row, col)?;
            let p2 = src2.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src1.channels() {
                pd[ch] = p1[ch] & p2[ch];
            }
        }
    }

    Ok(())
}

/// Bitwise OR operation
pub fn bitwise_or(src1: &Mat, src2: &Mat, dst: &mut Mat) -> Result<()> {
    if src1.rows() != src2.rows() || src1.cols() != src2.cols() {
        return Err(Error::InvalidDimensions(
            "Matrices must have same dimensions".to_string(),
        ));
    }

    *dst = Mat::new(src1.rows(), src1.cols(), src1.channels(), src1.depth())?;

    for row in 0..src1.rows() {
        for col in 0..src1.cols() {
            let p1 = src1.at(row, col)?;
            let p2 = src2.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src1.channels() {
                pd[ch] = p1[ch] | p2[ch];
            }
        }
    }

    Ok(())
}

/// Bitwise XOR operation
pub fn bitwise_xor(src1: &Mat, src2: &Mat, dst: &mut Mat) -> Result<()> {
    if src1.rows() != src2.rows() || src1.cols() != src2.cols() {
        return Err(Error::InvalidDimensions(
            "Matrices must have same dimensions".to_string(),
        ));
    }

    *dst = Mat::new(src1.rows(), src1.cols(), src1.channels(), src1.depth())?;

    for row in 0..src1.rows() {
        for col in 0..src1.cols() {
            let p1 = src1.at(row, col)?;
            let p2 = src2.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src1.channels() {
                pd[ch] = p1[ch] ^ p2[ch];
            }
        }
    }

    Ok(())
}

/// Bitwise NOT operation
pub fn bitwise_not(src: &Mat, dst: &mut Mat) -> Result<()> {
    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let ps = src.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src.channels() {
                pd[ch] = !ps[ch];
            }
        }
    }

    Ok(())
}

/// Split multi-channel matrix into separate single-channel matrices
pub fn split(src: &Mat) -> Result<Vec<Mat>> {
    let mut channels = Vec::new();

    for ch_idx in 0..src.channels() {
        let mut channel = Mat::new(src.rows(), src.cols(), 1, src.depth())?;

        for row in 0..src.rows() {
            for col in 0..src.cols() {
                let src_pixel = src.at(row, col)?;
                let dst_pixel = channel.at_mut(row, col)?;
                dst_pixel[0] = src_pixel[ch_idx];
            }
        }

        channels.push(channel);
    }

    Ok(channels)
}

/// Merge several single-channel matrices into one multi-channel matrix
pub fn merge(channels: &[Mat], dst: &mut Mat) -> Result<()> {
    if channels.is_empty() {
        return Err(Error::InvalidParameter(
            "At least one channel required".to_string(),
        ));
    }

    let rows = channels[0].rows();
    let cols = channels[0].cols();
    let depth = channels[0].depth();

    for ch in channels {
        if ch.rows() != rows || ch.cols() != cols {
            return Err(Error::InvalidDimensions(
                "All channels must have same dimensions".to_string(),
            ));
        }
        if ch.channels() != 1 {
            return Err(Error::InvalidParameter(
                "Input matrices must be single-channel".to_string(),
            ));
        }
    }

    *dst = Mat::new(rows, cols, channels.len(), depth)?;

    for row in 0..rows {
        for col in 0..cols {
            let dst_pixel = dst.at_mut(row, col)?;

            for (ch_idx, channel) in channels.iter().enumerate() {
                let src_pixel = channel.at(row, col)?;
                dst_pixel[ch_idx] = src_pixel[0];
            }
        }
    }

    Ok(())
}

/// Calculate mean value of matrix
pub fn mean(src: &Mat) -> Result<Scalar> {
    let mut sums = vec![0.0; src.channels().min(4)];
    let mut count = 0;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let pixel = src.at(row, col)?;
            for ch in 0..src.channels().min(4) {
                sums[ch] += f64::from(pixel[ch]);
            }
            count += 1;
        }
    }

    let mut result = [0.0; 4];
    for (i, sum) in sums.iter().enumerate() {
        result[i] = sum / f64::from(count);
    }

    Ok(Scalar { val: result })
}

/// Find minimum and maximum values and their locations
pub fn min_max_loc(src: &Mat) -> Result<(f64, f64, (usize, usize), (usize, usize))> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "min_max_loc only works on single-channel images".to_string(),
        ));
    }

    let mut min_val = 255.0;
    let mut max_val = 0.0;
    let mut min_loc = (0, 0);
    let mut max_loc = (0, 0);

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let pixel = src.at(row, col)?;
            let val = f64::from(pixel[0]);

            if val < min_val {
                min_val = val;
                min_loc = (row, col);
            }
            if val > max_val {
                max_val = val;
                max_loc = (row, col);
            }
        }
    }

    Ok((min_val, max_val, min_loc, max_loc))
}

/// Invert matrix element-wise
pub fn abs_diff(src1: &Mat, src2: &Mat, dst: &mut Mat) -> Result<()> {
    if src1.rows() != src2.rows() || src1.cols() != src2.cols() {
        return Err(Error::InvalidDimensions(
            "Matrices must have same dimensions".to_string(),
        ));
    }

    *dst = Mat::new(src1.rows(), src1.cols(), src1.channels(), src1.depth())?;

    for row in 0..src1.rows() {
        for col in 0..src1.cols() {
            let p1 = src1.at(row, col)?;
            let p2 = src2.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src1.channels() {
                // Use unsigned_abs to avoid sign loss warning
                let diff = i16::from(p1[ch]) - i16::from(p2[ch]);
                // unsigned_abs() returns u16, need to convert to u8
                let abs_diff = diff.unsigned_abs();
                // Clamp to u8 range (should already fit, but be safe)
                pd[ch] = abs_diff.min(255) as u8;
            }
        }
    }

    Ok(())
}

/// Apply look-up table to image
///
/// Transforms source image using look-up table:
/// `dst(I) = lut(src(I))`
pub fn lut(src: &Mat, lut_table: &Mat, dst: &mut Mat) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "lut only supports U8 depth".to_string(),
        ));
    }

    if lut_table.depth() != MatDepth::U8 {
        return Err(Error::InvalidParameter(
            "LUT table must be U8 depth".to_string(),
        ));
    }

    // LUT table should have 256 elements per channel
    let expected_size = 256 * src.channels();
    if lut_table.total() < expected_size {
        return Err(Error::InvalidParameter(format!(
            "LUT table too small: expected at least {} elements, got {}",
            expected_size,
            lut_table.total()
        )));
    }

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    let lut_data = lut_table.data();

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let src_pixel = src.at(row, col)?;
            let dst_pixel = dst.at_mut(row, col)?;

            for ch in 0..src.channels() {
                let src_val = src_pixel[ch];
                let lut_idx = usize::from(src_val) + ch * 256;
                dst_pixel[ch] = lut_data[lut_idx];
            }
        }
    }

    Ok(())
}

/// Normalize image to range [alpha, beta]
///
/// Scales image values to fit in specified range:
/// `dst = (src - min) / (max - min) * (beta - alpha) + alpha`
pub fn normalize(src: &Mat, dst: &mut Mat, alpha: f64, beta: f64) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "normalize only supports U8 depth".to_string(),
        ));
    }

    // Find min and max values
    let mut min_val = 255.0;
    let mut max_val = 0.0;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let pixel = src.at(row, col)?;
            for ch in 0..src.channels() {
                let val = f64::from(pixel[ch]);
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }
        }
    }

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    // Avoid division by zero
    if (max_val - min_val).abs() < 1e-10 {
        // All values are the same, set to alpha
        for row in 0..src.rows() {
            for col in 0..src.cols() {
                let dst_pixel = dst.at_mut(row, col)?;
                let clamped = alpha.clamp(0.0, 255.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let val = clamped as u8;
                for ch in 0..src.channels() {
                    dst_pixel[ch] = val;
                }
            }
        }
        return Ok(());
    }

    // Normalize values
    let scale = (beta - alpha) / (max_val - min_val);

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let src_pixel = src.at(row, col)?;
            let dst_pixel = dst.at_mut(row, col)?;

            for ch in 0..src.channels() {
                let src_val = f64::from(src_pixel[ch]);
                let normalized = (src_val - min_val) * scale + alpha;
                let clamped = normalized.clamp(0.0, 255.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let val = clamped as u8;
                dst_pixel[ch] = val;
            }
        }
    }

    Ok(())
}

/// Add weighted - calculates weighted sum of two arrays
pub fn add_weighted(
    src1: &Mat,
    alpha: f64,
    src2: &Mat,
    beta: f64,
    gamma: f64,
    dst: &mut Mat,
) -> Result<()> {
    if src1.rows() != src2.rows() || src1.cols() != src2.cols() {
        return Err(Error::InvalidDimensions(
            "Matrices must have same dimensions".to_string(),
        ));
    }

    *dst = Mat::new(src1.rows(), src1.cols(), src1.channels(), src1.depth())?;

    for row in 0..src1.rows() {
        for col in 0..src1.cols() {
            let p1 = src1.at(row, col)?;
            let p2 = src2.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src1.channels() {
                let val = alpha * f64::from(p1[ch]) + beta * f64::from(p2[ch]) + gamma;
                let clamped = val.clamp(0.0, 255.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let byte_val = clamped as u8;
                pd[ch] = byte_val;
            }
        }
    }

    Ok(())
}

/// Convert scale absolute - scales, calculates absolute values and converts result
pub fn convert_scale_abs(src: &Mat, dst: &mut Mat, alpha: f64, beta: f64) -> Result<()> {
    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let ps = src.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src.channels() {
                let val = (alpha * f64::from(ps[ch]) + beta).abs();
                let clamped = val.clamp(0.0, 255.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let byte_val = clamped as u8;
                pd[ch] = byte_val;
            }
        }
    }

    Ok(())
}

/// Calculate exponential of every array element
pub fn exp(src: &Mat, dst: &mut Mat) -> Result<()> {
    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let ps = src.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src.channels() {
                let val = (f64::from(ps[ch]) / 255.0).exp() * 255.0;
                let clamped = val.clamp(0.0, 255.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let byte_val = clamped as u8;
                pd[ch] = byte_val;
            }
        }
    }

    Ok(())
}

/// Calculate natural logarithm of every array element
pub fn log(src: &Mat, dst: &mut Mat) -> Result<()> {
    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let ps = src.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src.channels() {
                let val = f64::from(ps[ch]) / 255.0;
                let result = if val > 0.0 { val.ln() * 255.0 } else { 0.0 };
                let clamped = result.clamp(0.0, 255.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let byte_val = clamped as u8;
                pd[ch] = byte_val;
            }
        }
    }

    Ok(())
}

/// Raise every array element to a power
pub fn pow(src: &Mat, power: f64, dst: &mut Mat) -> Result<()> {
    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let ps = src.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src.channels() {
                let val = (f64::from(ps[ch]) / 255.0).powf(power) * 255.0;
                let clamped = val.clamp(0.0, 255.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let byte_val = clamped as u8;
                pd[ch] = byte_val;
            }
        }
    }

    Ok(())
}

/// Calculate square root of every array element
pub fn sqrt(src: &Mat, dst: &mut Mat) -> Result<()> {
    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let ps = src.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src.channels() {
                let val = (f64::from(ps[ch]) / 255.0).sqrt() * 255.0;
                let clamped = val.clamp(0.0, 255.0);
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let byte_val = clamped as u8;
                pd[ch] = byte_val;
            }
        }
    }

    Ok(())
}

/// Compute per-element minimum of two matrices
pub fn min(src1: &Mat, src2: &Mat, dst: &mut Mat) -> Result<()> {
    if src1.rows() != src2.rows() || src1.cols() != src2.cols() {
        return Err(Error::InvalidDimensions(
            "Matrices must have same dimensions".to_string(),
        ));
    }

    if src1.depth() != MatDepth::U8 || src2.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "min only supports U8 depth".to_string(),
        ));
    }

    *dst = Mat::new(src1.rows(), src1.cols(), src1.channels(), src1.depth())?;

    for row in 0..src1.rows() {
        for col in 0..src1.cols() {
            let p1 = src1.at(row, col)?;
            let p2 = src2.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src1.channels() {
                pd[ch] = p1[ch].min(p2[ch]);
            }
        }
    }

    Ok(())
}

/// Compute per-element maximum of two matrices
pub fn max(src1: &Mat, src2: &Mat, dst: &mut Mat) -> Result<()> {
    if src1.rows() != src2.rows() || src1.cols() != src2.cols() {
        return Err(Error::InvalidDimensions(
            "Matrices must have same dimensions".to_string(),
        ));
    }

    if src1.depth() != MatDepth::U8 || src2.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "max only supports U8 depth".to_string(),
        ));
    }

    *dst = Mat::new(src1.rows(), src1.cols(), src1.channels(), src1.depth())?;

    for row in 0..src1.rows() {
        for col in 0..src1.cols() {
            let p1 = src1.at(row, col)?;
            let p2 = src2.at(row, col)?;
            let pd = dst.at_mut(row, col)?;

            for ch in 0..src1.channels() {
                pd[ch] = p1[ch].max(p2[ch]);
            }
        }
    }

    Ok(())
}

/// Check if array elements lie between elements of two other arrays
pub fn in_range(src: &Mat, dst: &mut Mat, lower_bound: Scalar, upper_bound: Scalar) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "in_range only supports U8 depth".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let src_pixel = src.at(row, col)?;
            let dst_pixel = dst.at_mut(row, col)?;

            let mut in_range = true;
            for ch in 0..src.channels().min(4) {
                let val = f64::from(src_pixel[ch]);
                let lower = lower_bound.val[ch];
                let upper = upper_bound.val[ch];
                if val < lower || val > upper {
                    in_range = false;
                    break;
                }
            }

            dst_pixel[0] = if in_range { 255 } else { 0 };
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let src1 = Mat::new_with_default(10, 10, 3, MatDepth::U8, Scalar::all(100.0)).unwrap();
        let src2 = Mat::new_with_default(10, 10, 3, MatDepth::U8, Scalar::all(50.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        add(&src1, &src2, &mut dst).unwrap();

        let pixel = dst.at(5, 5).unwrap();
        assert_eq!(pixel[0], 150);
    }

    #[test]
    fn test_split_merge() {
        let src = Mat::new_with_default(10, 10, 3, MatDepth::U8, Scalar::from_rgb(255, 128, 64)).unwrap();

        let channels = split(&src).unwrap();
        assert_eq!(channels.len(), 3);

        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
        merge(&channels, &mut dst).unwrap();

        assert_eq!(dst.channels(), 3);
    }
}
