use crate::core::{Mat, MatDepth};
use crate::core::types::{Size, InterpolationFlag};
use crate::error::{Error, Result};

/// Resize an image
pub fn resize(src: &Mat, dst: &mut Mat, dsize: Size, interpolation: InterpolationFlag) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "resize only supports U8 depth".to_string(),
        ));
    }

    if dsize.width <= 0 || dsize.height <= 0 {
        return Err(Error::InvalidDimensions(
            "Destination size must be positive".to_string(),
        ));
    }

    let new_rows = dsize.height as usize;
    let new_cols = dsize.width as usize;

    *dst = Mat::new(new_rows, new_cols, src.channels(), src.depth())?;

    match interpolation {
        InterpolationFlag::Nearest => resize_nearest(src, dst),
        InterpolationFlag::Linear => resize_bilinear(src, dst),
        _ => Err(Error::UnsupportedOperation(format!(
            "Interpolation method {:?} not yet implemented",
            interpolation
        ))),
    }
}

/// Nearest neighbor interpolation
fn resize_nearest(src: &Mat, dst: &mut Mat) -> Result<()> {
    let x_ratio = src.cols() as f32 / dst.cols() as f32;
    let y_ratio = src.rows() as f32 / dst.rows() as f32;

    for dst_row in 0..dst.rows() {
        for dst_col in 0..dst.cols() {
            let src_row = (dst_row as f32 * y_ratio) as usize;
            let src_col = (dst_col as f32 * x_ratio) as usize;

            let src_row = src_row.min(src.rows() - 1);
            let src_col = src_col.min(src.cols() - 1);

            let src_pixel = src.at(src_row, src_col)?;
            let dst_pixel = dst.at_mut(dst_row, dst_col)?;

            dst_pixel.copy_from_slice(src_pixel);
        }
    }

    Ok(())
}

/// Bilinear interpolation
fn resize_bilinear(src: &Mat, dst: &mut Mat) -> Result<()> {
    let x_ratio = (src.cols() - 1) as f32 / dst.cols() as f32;
    let y_ratio = (src.rows() - 1) as f32 / dst.rows() as f32;

    for dst_row in 0..dst.rows() {
        for dst_col in 0..dst.cols() {
            let src_x = dst_col as f32 * x_ratio;
            let src_y = dst_row as f32 * y_ratio;

            let x1 = src_x.floor() as usize;
            let y1 = src_y.floor() as usize;
            let x2 = (x1 + 1).min(src.cols() - 1);
            let y2 = (y1 + 1).min(src.rows() - 1);

            let dx = src_x - x1 as f32;
            let dy = src_y - y1 as f32;

            let p11 = src.at(y1, x1)?;
            let p12 = src.at(y2, x1)?;
            let p21 = src.at(y1, x2)?;
            let p22 = src.at(y2, x2)?;

            let dst_pixel = dst.at_mut(dst_row, dst_col)?;

            for ch in 0..src.channels() {
                let v11 = p11[ch] as f32;
                let v12 = p12[ch] as f32;
                let v21 = p21[ch] as f32;
                let v22 = p22[ch] as f32;

                let v1 = v11 * (1.0 - dx) + v21 * dx;
                let v2 = v12 * (1.0 - dx) + v22 * dx;
                let v = v1 * (1.0 - dy) + v2 * dy;

                dst_pixel[ch] = v.round() as u8;
            }
        }
    }

    Ok(())
}

/// Flip an image
pub fn flip(src: &Mat, dst: &mut Mat, flip_code: i32) -> Result<()> {
    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    match flip_code {
        0 => {
            // Flip vertically
            for row in 0..src.rows() {
                for col in 0..src.cols() {
                    let src_pixel = src.at(row, col)?;
                    let dst_pixel = dst.at_mut(src.rows() - 1 - row, col)?;
                    dst_pixel.copy_from_slice(src_pixel);
                }
            }
        }
        1 => {
            // Flip horizontally
            for row in 0..src.rows() {
                for col in 0..src.cols() {
                    let src_pixel = src.at(row, col)?;
                    let dst_pixel = dst.at_mut(row, src.cols() - 1 - col)?;
                    dst_pixel.copy_from_slice(src_pixel);
                }
            }
        }
        -1 => {
            // Flip both
            for row in 0..src.rows() {
                for col in 0..src.cols() {
                    let src_pixel = src.at(row, col)?;
                    let dst_pixel = dst.at_mut(src.rows() - 1 - row, src.cols() - 1 - col)?;
                    dst_pixel.copy_from_slice(src_pixel);
                }
            }
        }
        _ => {
            return Err(Error::InvalidParameter(
                "flip_code must be 0 (vertical), 1 (horizontal), or -1 (both)".to_string(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_resize_nearest() {
        let src = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        resize(&src, &mut dst, Size::new(50, 50), InterpolationFlag::Nearest).unwrap();

        assert_eq!(dst.rows(), 50);
        assert_eq!(dst.cols(), 50);
    }

    #[test]
    fn test_flip() {
        let mut src = Mat::new(10, 10, 3, MatDepth::U8).unwrap();
        let pixel = src.at_mut(0, 0).unwrap();
        pixel[0] = 255;

        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
        flip(&src, &mut dst, 0).unwrap();

        let flipped = dst.at(9, 0).unwrap();
        assert_eq!(flipped[0], 255);
    }
}
