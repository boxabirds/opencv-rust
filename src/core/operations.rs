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
                let val = (p1[ch] as f64 * p2[ch] as f64 * scale).clamp(0.0, 255.0);
                pd[ch] = val as u8;
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
                sums[ch] += pixel[ch] as f64;
            }
            count += 1;
        }
    }

    let mut result = [0.0; 4];
    for (i, sum) in sums.iter().enumerate() {
        result[i] = sum / count as f64;
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
            let val = pixel[0] as f64;

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
                pd[ch] = (p1[ch] as i16 - p2[ch] as i16).abs() as u8;
            }
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
