use crate::core::{Mat, MatDepth};
use crate::core::types::{Size, InterpolationFlag, Point2f};
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

/// Warp affine transformation
pub fn warp_affine(
    src: &Mat,
    dst: &mut Mat,
    m: &[[f64; 3]; 2],
    dsize: Size,
) -> Result<()> {
    *dst = Mat::new(dsize.height as usize, dsize.width as usize, src.channels(), src.depth())?;

    for row in 0..dst.rows() {
        for col in 0..dst.cols() {
            let x = col as f64;
            let y = row as f64;

            // Apply transformation
            let src_x = m[0][0] * x + m[0][1] * y + m[0][2];
            let src_y = m[1][0] * x + m[1][1] * y + m[1][2];

            let src_col = src_x as i32;
            let src_row = src_y as i32;

            if src_row >= 0 && src_row < src.rows() as i32
                && src_col >= 0 && src_col < src.cols() as i32
            {
                let src_pixel = src.at(src_row as usize, src_col as usize)?;
                let dst_pixel = dst.at_mut(row, col)?;
                dst_pixel.copy_from_slice(src_pixel);
            }
        }
    }

    Ok(())
}

/// Warp perspective transformation
pub fn warp_perspective(
    src: &Mat,
    dst: &mut Mat,
    m: &[[f64; 3]; 3],
    dsize: Size,
) -> Result<()> {
    *dst = Mat::new(dsize.height as usize, dsize.width as usize, src.channels(), src.depth())?;

    for row in 0..dst.rows() {
        for col in 0..dst.cols() {
            let x = col as f64;
            let y = row as f64;

            // Apply homography
            let w = m[2][0] * x + m[2][1] * y + m[2][2];

            if w.abs() < 1e-10 {
                continue;
            }

            let src_x = (m[0][0] * x + m[0][1] * y + m[0][2]) / w;
            let src_y = (m[1][0] * x + m[1][1] * y + m[1][2]) / w;

            let src_col = src_x as i32;
            let src_row = src_y as i32;

            if src_row >= 0 && src_row < src.rows() as i32
                && src_col >= 0 && src_col < src.cols() as i32
            {
                let src_pixel = src.at(src_row as usize, src_col as usize)?;
                let dst_pixel = dst.at_mut(row, col)?;
                dst_pixel.copy_from_slice(src_pixel);
            }
        }
    }

    Ok(())
}

/// Get rotation matrix for 2D rotation
pub fn get_rotation_matrix_2d(center: Point2f, angle: f64, scale: f64) -> [[f64; 3]; 2] {
    let alpha = scale * angle.to_radians().cos();
    let beta = scale * angle.to_radians().sin();

    [
        [alpha, beta, (1.0 - alpha) * center.x as f64 - beta * center.y as f64],
        [-beta, alpha, beta * center.x as f64 + (1.0 - alpha) * center.y as f64],
    ]
}

/// Get affine transformation from three point pairs
pub fn get_affine_transform(src: &[Point2f; 3], dst: &[Point2f; 3]) -> [[f64; 3]; 2] {
    let x0 = src[0].x as f64;
    let y0 = src[0].y as f64;
    let x1 = src[1].x as f64;
    let y1 = src[1].y as f64;
    let x2 = src[2].x as f64;
    let y2 = src[2].y as f64;

    let u0 = dst[0].x as f64;
    let v0 = dst[0].y as f64;
    let u1 = dst[1].x as f64;
    let v1 = dst[1].y as f64;
    let u2 = dst[2].x as f64;
    let v2 = dst[2].y as f64;

    let det = x0 * (y1 - y2) - x1 * (y0 - y2) + x2 * (y0 - y1);

    if det.abs() < 1e-10 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    }

    let a11 = (u0 * (y1 - y2) - u1 * (y0 - y2) + u2 * (y0 - y1)) / det;
    let a12 = (x0 * (u2 - u1) - x1 * (u2 - u0) + x2 * (u1 - u0)) / det;
    let a13 = (x0 * (y1 * u2 - y2 * u1) - x1 * (y0 * u2 - y2 * u0) + x2 * (y0 * u1 - y1 * u0)) / det;

    let a21 = (v0 * (y1 - y2) - v1 * (y0 - y2) + v2 * (y0 - y1)) / det;
    let a22 = (x0 * (v2 - v1) - x1 * (v2 - v0) + x2 * (v1 - v0)) / det;
    let a23 = (x0 * (y1 * v2 - y2 * v1) - x1 * (y0 * v2 - y2 * v0) + x2 * (y0 * v1 - y1 * v0)) / det;

    [[a11, a12, a13], [a21, a22, a23]]
}

/// Rotate image by 90, 180, or 270 degrees
pub fn rotate(src: &Mat, dst: &mut Mat, rotate_code: RotateCode) -> Result<()> {
    match rotate_code {
        RotateCode::Rotate90Clockwise => {
            *dst = Mat::new(src.cols(), src.rows(), src.channels(), src.depth())?;
            for row in 0..src.rows() {
                for col in 0..src.cols() {
                    let dst_row = col;
                    let dst_col = src.rows() - 1 - row;
                    let src_pixel = src.at(row, col)?;
                    let dst_pixel = dst.at_mut(dst_row, dst_col)?;
                    dst_pixel.copy_from_slice(src_pixel);
                }
            }
        }
        RotateCode::Rotate180 => {
            *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;
            for row in 0..src.rows() {
                for col in 0..src.cols() {
                    let dst_row = src.rows() - 1 - row;
                    let dst_col = src.cols() - 1 - col;
                    let src_pixel = src.at(row, col)?;
                    let dst_pixel = dst.at_mut(dst_row, dst_col)?;
                    dst_pixel.copy_from_slice(src_pixel);
                }
            }
        }
        RotateCode::Rotate90CounterClockwise => {
            *dst = Mat::new(src.cols(), src.rows(), src.channels(), src.depth())?;
            for row in 0..src.rows() {
                for col in 0..src.cols() {
                    let dst_row = src.cols() - 1 - col;
                    let dst_col = row;
                    let src_pixel = src.at(row, col)?;
                    let dst_pixel = dst.at_mut(dst_row, dst_col)?;
                    dst_pixel.copy_from_slice(src_pixel);
                }
            }
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub enum RotateCode {
    Rotate90Clockwise,
    Rotate180,
    Rotate90CounterClockwise,
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

    #[test]
    fn test_warp_affine() {
        let src = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        // Identity transformation
        let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

        warp_affine(&src, &mut dst, &m, Size::new(100, 100)).unwrap();

        assert_eq!(dst.rows(), 100);
        assert_eq!(dst.cols(), 100);
    }

    #[test]
    fn test_get_rotation_matrix_2d() {
        let center = Point2f::new(50.0, 50.0);
        let m = get_rotation_matrix_2d(center, 45.0, 1.0);

        // Check that it's a valid matrix
        assert!(m[0][0].abs() > 0.0);
    }

    #[test]
    fn test_rotate() {
        let src = Mat::new_with_default(50, 100, 3, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

        rotate(&src, &mut dst, RotateCode::Rotate90Clockwise).unwrap();

        assert_eq!(dst.rows(), 100);
        assert_eq!(dst.cols(), 50);
    }
}
