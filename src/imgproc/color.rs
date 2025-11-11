use crate::core::{Mat, MatDepth};
use crate::core::types::ColorConversionCode;
use crate::error::{Error, Result};

/// Convert color space of an image with GPU acceleration (async for WASM)
pub async fn cvt_color_async(
    src: &Mat,
    dst: &mut Mat,
    code: ColorConversionCode,
    use_gpu: bool,
) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "cvt_color only supports U8 depth".to_string(),
        ));
    }

    // Try GPU if requested and available
    if use_gpu {
        #[cfg(feature = "gpu")]
        {
            match code {
                ColorConversionCode::RgbToGray => {
                    use crate::gpu::ops::rgb_to_gray_gpu_async;
                    match rgb_to_gray_gpu_async(src, dst).await {
                        Ok(()) => return Ok(()),
                        Err(_) => { /* Fall through to CPU */ }
                    }
                }
                ColorConversionCode::RgbToHsv => {
                    use crate::gpu::ops::rgb_to_hsv_gpu_async;
                    match rgb_to_hsv_gpu_async(src, dst).await {
                        Ok(()) => return Ok(()),
                        Err(_) => { /* Fall through to CPU */ }
                    }
                }
                _ => { /* Fall through to CPU for unsupported GPU conversions */ }
            }
        }
    }

    // CPU fallback
    cvt_color(src, dst, code)
}

/// Convert color space of an image (CPU-only, sync)
pub fn cvt_color(src: &Mat, dst: &mut Mat, code: ColorConversionCode) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "cvt_color only supports U8 depth".to_string(),
        ));
    }

    // GPU acceleration handled by individual conversion functions
    // (rgb_to_gray_gpu, rgb_to_hsv_gpu, etc.) - no generic cvt_color_gpu

    // CPU fallback
    match code {
        ColorConversionCode::BgrToGray | ColorConversionCode::RgbToGray => {
            bgr_to_gray(src, dst, code == ColorConversionCode::BgrToGray)
        }
        ColorConversionCode::BgraToGray | ColorConversionCode::RgbaToGray => {
            rgba_to_gray(src, dst, code == ColorConversionCode::BgraToGray)
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
        ColorConversionCode::BgrToLab | ColorConversionCode::RgbToLab => {
            rgb_to_lab(src, dst, code == ColorConversionCode::BgrToLab)
        }
        ColorConversionCode::LabToBgr | ColorConversionCode::LabToRgb => {
            lab_to_rgb(src, dst, code == ColorConversionCode::LabToBgr)
        }
        ColorConversionCode::BgrToYCrCb | ColorConversionCode::RgbToYCrCb => {
            rgb_to_ycrcb(src, dst, code == ColorConversionCode::BgrToYCrCb)
        }
        ColorConversionCode::YCrCbToBgr | ColorConversionCode::YCrCbToRgb => {
            ycrcb_to_rgb(src, dst, code == ColorConversionCode::YCrCbToBgr)
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

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let cols = src.cols();
        let channels = src.channels();

        dst.data_mut().par_chunks_mut(cols).enumerate().for_each(|(row, dst_row)| {
            for (col, dst_pixel) in dst_row.iter_mut().enumerate() {
                let src_idx = (row * cols + col) * channels;
                let src_data = src.data();
                let (r, g, b) = if is_bgr {
                    (src_data[src_idx + 2], src_data[src_idx + 1], src_data[src_idx])
                } else {
                    (src_data[src_idx], src_data[src_idx + 1], src_data[src_idx + 2])
                };

                // Using standard RGB to grayscale conversion weights
                let gray = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;
                *dst_pixel = gray;
            }
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
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
    }

    Ok(())
}

/// Convert RGBA/BGRA to grayscale (ignoring alpha channel)
fn rgba_to_gray(src: &Mat, dst: &mut Mat, is_bgra: bool) -> Result<()> {
    if src.channels() != 4 {
        return Err(Error::InvalidParameter(
            "Source must have 4 channels".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    #[cfg(feature = "rayon")]
    {
        use rayon::prelude::*;
        let cols = src.cols();
        let channels = src.channels();

        dst.data_mut().par_chunks_mut(cols).enumerate().for_each(|(row, dst_row)| {
            for (col, dst_pixel) in dst_row.iter_mut().enumerate() {
                let src_idx = (row * cols + col) * channels;
                let src_data = src.data();
                let (r, g, b) = if is_bgra {
                    (src_data[src_idx + 2], src_data[src_idx + 1], src_data[src_idx])
                } else {
                    (src_data[src_idx], src_data[src_idx + 1], src_data[src_idx + 2])
                };

                let gray = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;
                *dst_pixel = gray;
            }
        });
    }

    #[cfg(not(feature = "rayon"))]
    {
        for row in 0..src.rows() {
            for col in 0..src.cols() {
                let pixel = src.at(row, col)?;
                let (r, g, b) = if is_bgra {
                    (pixel[2], pixel[1], pixel[0])
                } else {
                    (pixel[0], pixel[1], pixel[2])
                };

                let gray = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32) as u8;

                let dst_pixel = dst.at_mut(row, col)?;
                dst_pixel[0] = gray;
            }
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

/// Convert RGB/BGR to Lab color space
fn rgb_to_lab(src: &Mat, dst: &mut Mat, is_bgr: bool) -> Result<()> {
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

            // Convert to XYZ (D65 illuminant)
            let r_linear = if r > 0.04045 { ((r + 0.055) / 1.055).powf(2.4) } else { r / 12.92 };
            let g_linear = if g > 0.04045 { ((g + 0.055) / 1.055).powf(2.4) } else { g / 12.92 };
            let b_linear = if b > 0.04045 { ((b + 0.055) / 1.055).powf(2.4) } else { b / 12.92 };

            let x = r_linear * 0.4124 + g_linear * 0.3576 + b_linear * 0.1805;
            let y = r_linear * 0.2126 + g_linear * 0.7152 + b_linear * 0.0722;
            let z = r_linear * 0.0193 + g_linear * 0.1192 + b_linear * 0.9505;

            // Normalize for D65
            let xn = x / 0.950489;
            let yn = y / 1.0;
            let zn = z / 1.08884;

            // Convert to Lab
            let f = |t: f32| if t > 0.008856 { t.powf(1.0 / 3.0) } else { 7.787 * t + 16.0 / 116.0 };
            let fx = f(xn);
            let fy = f(yn);
            let fz = f(zn);

            let l = 116.0 * fy - 16.0;
            let a = 500.0 * (fx - fy);
            let b_lab = 200.0 * (fy - fz);

            let dst_pixel = dst.at_mut(row, col)?;
            dst_pixel[0] = (l * 2.55).clamp(0.0, 255.0) as u8;  // L in [0, 255]
            dst_pixel[1] = (a + 128.0).clamp(0.0, 255.0) as u8;  // a in [0, 255]
            dst_pixel[2] = (b_lab + 128.0).clamp(0.0, 255.0) as u8;  // b in [0, 255]
        }
    }

    Ok(())
}

/// Convert Lab to RGB/BGR
fn lab_to_rgb(src: &Mat, dst: &mut Mat, is_bgr: bool) -> Result<()> {
    if src.channels() != 3 {
        return Err(Error::InvalidParameter(
            "Source must have 3 channels".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 3, MatDepth::U8)?;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let pixel = src.at(row, col)?;
            let l = pixel[0] as f32 / 2.55;
            let a = pixel[1] as f32 - 128.0;
            let b = pixel[2] as f32 - 128.0;

            // Convert to XYZ
            let fy = (l + 16.0) / 116.0;
            let fx = a / 500.0 + fy;
            let fz = fy - b / 200.0;

            let f_inv = |t: f32| if t > 0.206897 { t.powi(3) } else { (t - 16.0 / 116.0) / 7.787 };
            let xn = f_inv(fx) * 0.950489;
            let yn = f_inv(fy) * 1.0;
            let zn = f_inv(fz) * 1.08884;

            // Convert to RGB
            let r_linear = xn * 3.2406 + yn * -1.5372 + zn * -0.4986;
            let g_linear = xn * -0.9689 + yn * 1.8758 + zn * 0.0415;
            let b_linear = xn * 0.0557 + yn * -0.2040 + zn * 1.0570;

            let gamma = |t: f32| if t > 0.0031308 { 1.055 * t.powf(1.0 / 2.4) - 0.055 } else { 12.92 * t };
            let r = (gamma(r_linear) * 255.0).clamp(0.0, 255.0) as u8;
            let g = (gamma(g_linear) * 255.0).clamp(0.0, 255.0) as u8;
            let b_rgb = (gamma(b_linear) * 255.0).clamp(0.0, 255.0) as u8;

            let dst_pixel = dst.at_mut(row, col)?;
            if is_bgr {
                dst_pixel[0] = b_rgb;
                dst_pixel[1] = g;
                dst_pixel[2] = r;
            } else {
                dst_pixel[0] = r;
                dst_pixel[1] = g;
                dst_pixel[2] = b_rgb;
            }
        }
    }

    Ok(())
}

/// Convert RGB/BGR to YCrCb
fn rgb_to_ycrcb(src: &Mat, dst: &mut Mat, is_bgr: bool) -> Result<()> {
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
                (pixel[2] as f32, pixel[1] as f32, pixel[0] as f32)
            } else {
                (pixel[0] as f32, pixel[1] as f32, pixel[2] as f32)
            };

            // ITU-R BT.601 conversion
            let y = 0.299 * r + 0.587 * g + 0.114 * b;
            let cr = (r - y) * 0.713 + 128.0;
            let cb = (b - y) * 0.564 + 128.0;

            let dst_pixel = dst.at_mut(row, col)?;
            dst_pixel[0] = y.clamp(0.0, 255.0) as u8;
            dst_pixel[1] = cr.clamp(0.0, 255.0) as u8;
            dst_pixel[2] = cb.clamp(0.0, 255.0) as u8;
        }
    }

    Ok(())
}

/// Convert YCrCb to RGB/BGR
fn ycrcb_to_rgb(src: &Mat, dst: &mut Mat, is_bgr: bool) -> Result<()> {
    if src.channels() != 3 {
        return Err(Error::InvalidParameter(
            "Source must have 3 channels".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 3, MatDepth::U8)?;

    for row in 0..src.rows() {
        for col in 0..src.cols() {
            let pixel = src.at(row, col)?;
            let y = pixel[0] as f32;
            let cr = pixel[1] as f32 - 128.0;
            let cb = pixel[2] as f32 - 128.0;

            // ITU-R BT.601 conversion
            let r = y + 1.403 * cr;
            let g = y - 0.714 * cr - 0.344 * cb;
            let b = y + 1.773 * cb;

            let dst_pixel = dst.at_mut(row, col)?;
            if is_bgr {
                dst_pixel[0] = b.clamp(0.0, 255.0) as u8;
                dst_pixel[1] = g.clamp(0.0, 255.0) as u8;
                dst_pixel[2] = r.clamp(0.0, 255.0) as u8;
            } else {
                dst_pixel[0] = r.clamp(0.0, 255.0) as u8;
                dst_pixel[1] = g.clamp(0.0, 255.0) as u8;
                dst_pixel[2] = b.clamp(0.0, 255.0) as u8;
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
