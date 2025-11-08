use crate::core::{Mat, MatDepth};
use crate::core::types::ColorConversionCode;
use crate::error::{Error, Result};

/// Convert color space of an image
pub fn cvt_color(src: &Mat, dst: &mut Mat, code: ColorConversionCode) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "cvt_color only supports U8 depth".to_string(),
        ));
    }

    match code {
        ColorConversionCode::BgrToGray | ColorConversionCode::RgbToGray => {
            bgr_to_gray(src, dst, code == ColorConversionCode::BgrToGray)
        }
        ColorConversionCode::GrayToBgr | ColorConversionCode::GrayToRgb => {
            gray_to_bgr(src, dst)
        }
        ColorConversionCode::BgrToRgb | ColorConversionCode::RgbToBgr => {
            swap_rb_channels(src, dst)
        }
        ColorConversionCode::BgrToHsv | ColorConversionCode::RgbToHsv => {
            rgb_to_hsv(src, dst, code == ColorConversionCode::BgrToHsv)
        }
        ColorConversionCode::HsvToBgr | ColorConversionCode::HsvToRgb => {
            hsv_to_rgb(src, dst, code == ColorConversionCode::HsvToBgr)
        }
    }
}

/// Convert BGR/RGB to grayscale
fn bgr_to_gray(src: &Mat, dst: &mut Mat, is_bgr: bool) -> Result<()> {
    if src.channels() != 3 {
        return Err(Error::InvalidParameter(
            "Source must have 3 channels".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let pixel = src.at(row, col)?;
            let (r, g, b) = if is_bgr {
                (pixel[2], pixel[1], pixel[0])
            } else {
                (pixel[0], pixel[1], pixel[2])
            };

            // Using standard RGB to grayscale conversion weights
            let gray = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;

            let dst_pixel = dst.at_mut(row, col)?;
            dst_pixel[0] = gray;
        }
    }

    Ok(())
}

/// Convert grayscale to BGR/RGB
fn gray_to_bgr(src: &Mat, dst: &mut Mat) -> Result<()> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Source must have 1 channel".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 3, MatDepth::U8)?;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let pixel = src.at(row, col)?;
            let gray = pixel[0];

            let dst_pixel = dst.at_mut(row, col)?;
            dst_pixel[0] = gray;
            dst_pixel[1] = gray;
            dst_pixel[2] = gray;
        }
    }

    Ok(())
}

/// Swap R and B channels (BGR <-> RGB)
fn swap_rb_channels(src: &Mat, dst: &mut Mat) -> Result<()> {
    if src.channels() != 3 {
        return Err(Error::InvalidParameter(
            "Source must have 3 channels".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 3, MatDepth::U8)?;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let pixel = src.at(row, col)?;
            let dst_pixel = dst.at_mut(row, col)?;

            dst_pixel[0] = pixel[2];
            dst_pixel[1] = pixel[1];
            dst_pixel[2] = pixel[0];
        }
    }

    Ok(())
}

/// Convert RGB/BGR to HSV
fn rgb_to_hsv(src: &Mat, dst: &mut Mat, is_bgr: bool) -> Result<()> {
    if src.channels() != 3 {
        return Err(Error::InvalidParameter(
            "Source must have 3 channels".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 3, MatDepth::U8)?;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let pixel = src.at(row, col)?;
            let (r, g, b) = if is_bgr {
                (pixel[2] as f32 / 255.0, pixel[1] as f32 / 255.0, pixel[0] as f32 / 255.0)
            } else {
                (pixel[0] as f32 / 255.0, pixel[1] as f32 / 255.0, pixel[2] as f32 / 255.0)
            };

            let max = r.max(g).max(b);
            let min = r.min(g).min(b);
            let delta = max - min;

            // Hue calculation
            let h = if delta == 0.0 {
                0.0
            } else if max == r {
                60.0 * (((g - b) / delta) % 6.0)
            } else if max == g {
                60.0 * (((b - r) / delta) + 2.0)
            } else {
                60.0 * (((r - g) / delta) + 4.0)
            };

            let h = if h < 0.0 { h + 360.0 } else { h };

            // Saturation calculation
            let s = if max == 0.0 { 0.0 } else { delta / max };

            // Value
            let v = max;

            let dst_pixel = dst.at_mut(row, col)?;
            dst_pixel[0] = (h / 2.0) as u8; // OpenCV stores H in range [0, 180]
            dst_pixel[1] = (s * 255.0) as u8;
            dst_pixel[2] = (v * 255.0) as u8;
        }
    }

    Ok(())
}

/// Convert HSV to RGB/BGR
fn hsv_to_rgb(src: &Mat, dst: &mut Mat, is_bgr: bool) -> Result<()> {
    if src.channels() != 3 {
        return Err(Error::InvalidParameter(
            "Source must have 3 channels".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 3, MatDepth::U8)?;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let pixel = src.at(row, col)?;
            let h = pixel[0] as f32 * 2.0; // Convert back from [0, 180] to [0, 360]
            let s = pixel[1] as f32 / 255.0;
            let v = pixel[2] as f32 / 255.0;

            let c = v * s;
            let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
            let m = v - c;

            let (r, g, b) = if h < 60.0 {
                (c, x, 0.0)
            } else if h < 120.0 {
                (x, c, 0.0)
            } else if h < 180.0 {
                (0.0, c, x)
            } else if h < 240.0 {
                (0.0, x, c)
            } else if h < 300.0 {
                (x, 0.0, c)
            } else {
                (c, 0.0, x)
            };

            let r = ((r + m) * 255.0) as u8;
            let g = ((g + m) * 255.0) as u8;
            let b = ((b + m) * 255.0) as u8;

            let dst_pixel = dst.at_mut(row, col)?;
            if is_bgr {
                dst_pixel[0] = b;
                dst_pixel[1] = g;
                dst_pixel[2] = r;
            } else {
                dst_pixel[0] = r;
                dst_pixel[1] = g;
                dst_pixel[2] = b;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_to_gray() {
        let mut src = Mat::new(10, 10, 3, MatDepth::U8).unwrap();
        // Set a red pixel
        let pixel = src.at_mut(5, 5).unwrap();
        pixel[0] = 255;
        pixel[1] = 0;
        pixel[2] = 0;

        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
        cvt_color(&src, &mut dst, ColorConversionCode::RgbToGray).unwrap();

        assert_eq!(dst.channels(), 1);
    }

    #[test]
    fn test_bgr_to_rgb() {
        let mut src = Mat::new(10, 10, 3, MatDepth::U8).unwrap();
        let pixel = src.at_mut(5, 5).unwrap();
        pixel[0] = 100;
        pixel[1] = 150;
        pixel[2] = 200;

        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
        cvt_color(&src, &mut dst, ColorConversionCode::BgrToRgb).unwrap();

        let result = dst.at(5, 5).unwrap();
        assert_eq!(result[0], 200);
        assert_eq!(result[1], 150);
        assert_eq!(result[2], 100);
    }
}
