//! WASM bindings

use wasm_bindgen::prelude::*;
use crate::core::{Mat, MatDepth};
use crate::wasm::WasmMat;

// ===== distanceTransform =====
#[wasm_bindgen(js_name = distanceTransform)]
pub async fn distance_transform_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::advanced_filter::distance_transform;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut dst = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::distance_transform_gpu_async(&gray, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            distance_transform(&gray, &mut dst, crate::imgproc::advanced_filter::DistanceType::L2, 3)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== houghLines =====
#[wasm_bindgen(js_name = houghLines)]
pub async fn hough_lines_wasm(
    src: &WasmMat,
    threshold: i32,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::line;
    use crate::core::types::{ColorConversionCode, Point, Scalar};
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let lines = crate::backend_dispatch! {
        gpu => {
            return Err(JsValue::from_str("GPU not implemented for hough_lines"));
        }
        cpu => {
            crate::imgproc::hough::hough_lines(&gray, 1.0, std::f64::consts::PI / 180.0, threshold)
                .map_err(|e| JsValue::from_str(&e.to_string()))?
        }
    };

    // Draw lines on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0); // Green

    for (rho, theta) in lines.iter().take(50) { // Limit to 50 lines
        let a = theta.cos();
        let b = theta.sin();
        let x0 = a * rho;
        let y0 = b * rho;
        let x1 = (x0 + 1000.0 * (-b)) as i32;
        let y1 = (y0 + 1000.0 * a) as i32;
        let x2 = (x0 - 1000.0 * (-b)) as i32;
        let y2 = (y0 - 1000.0 * a) as i32;

        let _ = line(&mut result, Point::new(x1, y1), Point::new(x2, y2), color, 2);
    }

    Ok(WasmMat { inner: result })
}


// ===== houghLinesP =====
#[wasm_bindgen(js_name = houghLinesP)]
pub async fn hough_lines_p_wasm(
    src: &WasmMat,
    threshold: i32,
    min_line_length: f64,
    max_line_gap: f64,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::hough::hough_lines_p;
    use crate::imgproc::drawing::line;
    use crate::core::types::{ColorConversionCode, Point, Scalar};
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let lines = hough_lines_p(&gray, 1.0, std::f64::consts::PI / 180.0, threshold, min_line_length, max_line_gap)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw line segments on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 0.0, 255.0, 255.0); // Red

    for (p1, p2) in lines {
        let _ = line(&mut result, p1, p2, color, 2);
    }

    Ok(WasmMat { inner: result })
}


// ===== houghCircles =====
#[wasm_bindgen(js_name = houghCircles)]
pub async fn hough_circles_wasm(
    src: &WasmMat,
    min_dist: f64,
    param1: f64,
    param2: f64,
    min_radius: i32,
    max_radius: i32,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::hough::{hough_circles, HoughCirclesMethod};
    use crate::imgproc::drawing::circle;
    use crate::core::types::{ColorConversionCode, Point, Scalar};
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let circles = hough_circles(&gray, HoughCirclesMethod::Gradient, 1.0, min_dist, param1, param2, min_radius, max_radius)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw circles on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 255.0, 255.0); // Magenta

    for c in circles {
        let _ = circle(&mut result, c.center, c.radius, color);
    }

    Ok(WasmMat { inner: result })
}


// ===== logFilter =====
#[wasm_bindgen(js_name = logFilter)]
pub async fn log_filter_wasm(src: &WasmMat, ksize: i32, sigma: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::advanced_filter::laplacian_of_gaussian;
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;

    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut dst = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    laplacian_of_gaussian(&gray, &mut dst, ksize, sigma)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


// ===== inpaint =====
#[wasm_bindgen(js_name = inpaint)]
pub async fn inpaint_wasm(src: &WasmMat, radius: i32) -> Result<WasmMat, JsValue> {
    use crate::photo::inpaint;

    // Create a mask (central region to inpaint)
    let mut mask = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Mark center region as damaged
    let cx = (src.inner.cols() / 2) as i32;
    let cy = (src.inner.rows() / 2) as i32;
    let r = (src.inner.cols().min(src.inner.rows()) / 4) as i32;

    for row in 0..mask.rows() {
        for col in 0..mask.cols() {
            let dx = col as i32 - cx;
            let dy = row as i32 - cy;
            let dist_sq = dx * dx + dy * dy;
            mask.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] =
                if dist_sq < (r * r) { 255 } else { 0 };
        }
    }

    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    inpaint(&src.inner, &mask, &mut dst, radius as f64)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


// ===== tonemapDrago =====
#[wasm_bindgen(js_name = tonemapDrago)]
pub async fn tonemap_drago_wasm(src: &WasmMat, bias: f64) -> Result<WasmMat, JsValue> {
    use crate::photo::hdr::TonemapDrago;

    let tonemap = TonemapDrago::new().with_bias(bias as f32);
    let dst = tonemap.process(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


// ===== tonemapReinhard =====
#[wasm_bindgen(js_name = tonemapReinhard)]
pub async fn tonemap_reinhard_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::photo::hdr::TonemapReinhard;

    let tonemap = TonemapReinhard::new();
    let dst = tonemap.process(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


// ===== bruteForceMatcher =====
#[wasm_bindgen(js_name = bruteForceMatcher)]
pub async fn brute_force_matcher_wasm(src: &WasmMat, n_features: usize) -> Result<WasmMat, JsValue> {
    use crate::features2d::SIFTF32;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{ColorConversionCode, Point, Scalar};

    // Convert to grayscale
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    // Detect features in left and right halves
    let mid = (gray.cols() / 2) as i32;
    let mut left_half = Mat::new(gray.rows(), mid as usize, 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut right_half = Mat::new(gray.rows(), gray.cols() - mid as usize, 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    for row in 0..gray.rows() {
        for col in 0..(mid as usize) {
            left_half.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] =
                gray.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0];
        }
        for col in (mid as usize)..gray.cols() {
            right_half.at_mut(row, col - mid as usize).map_err(|e| JsValue::from_str(&e.to_string()))?[0] =
                gray.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0];
        }
    }

    let sift = SIFTF32::new(n_features / 2);
    let (kp1, _) = sift.detect_and_compute(&left_half)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let (kp2, _) = sift.detect_and_compute(&right_half)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on left and right
    let mut result = src.inner.clone();
    for kp in kp1.iter().take(20) {
        let pt = Point::new(kp.pt.x as i32, kp.pt.y as i32);
        let _ = circle(&mut result, pt, 3, Scalar::new(0.0, 255.0, 0.0, 255.0));
    }
    for kp in kp2.iter().take(20) {
        let pt = Point::new((kp.pt.x as i32) + mid, kp.pt.y as i32);
        let _ = circle(&mut result, pt, 3, Scalar::new(255.0, 0.0, 0.0, 255.0));
    }

    Ok(WasmMat { inner: result })
}


// ===== superResolution =====
#[wasm_bindgen(js_name = superResolution)]
pub async fn super_resolution_wasm(src: &WasmMat, scale: f32) -> Result<WasmMat, JsValue> {
    use crate::photo::super_resolution::SuperResolutionBicubic;

    let sr = SuperResolutionBicubic::new(scale);
    let dst = sr.process(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


// ===== mergeDebevec =====
#[wasm_bindgen(js_name = mergeDebevec)]
pub async fn merge_debevec_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::photo::hdr::MergeDebevec;

    // For demo, use same image with different exposures (simulated)
    let images = vec![src.inner.clone()];
    let times = vec![1.0 / 30.0];

    let merge = MergeDebevec::new();
    let hdr = merge.process(&images, &times)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: hdr })
}


// ===== panoramaStitcher =====
#[wasm_bindgen(js_name = panoramaStitcher)]
pub async fn panorama_stitcher_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::put_text;
    use crate::core::types::{Point, Scalar};

    // For single image, just add annotation
    let mut result = src.inner.clone();
    let text = "Panorama stitching demo".to_string();
    let _ = put_text(&mut result, &text, Point::new(10, 30), 0.7, Scalar::new(255.0, 255.0, 0.0, 255.0));

    Ok(WasmMat { inner: result })
}


// ===== featherBlender =====
#[wasm_bindgen(js_name = featherBlender)]
pub async fn feather_blender_wasm(src: &WasmMat, blend_strength: f32) -> Result<WasmMat, JsValue> {
    // Simple alpha blending demo
    let mut result = src.inner.clone();
    
    // Apply feathering effect to edges
    for row in 0..result.rows() {
        for col in 0..result.cols() {
            let edge_dist = col.min(result.cols() - col).min(row).min(result.rows() - row) as f32;
            let alpha = (edge_dist * blend_strength).min(1.0);

            let num_channels = result.channels();
            let pixel = result.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
            for ch in 0..num_channels {
                pixel[ch] = (pixel[ch] as f32 * alpha) as u8;
            }
        }
    }

    Ok(WasmMat { inner: result })
}


// ===== multibandBlender =====
#[wasm_bindgen(js_name = multibandBlender)]
pub async fn multiband_blender_wasm(src: &WasmMat, num_bands: usize) -> Result<WasmMat, JsValue> {
    use crate::stitching::blending::MultiBandBlender;
    use crate::core::{Mat, MatDepth};
    
    // Create two overlapping images for blending demo
    let w = src.inner.cols();
    let h = src.inner.rows();
    
    // Split image into left and right halves with overlap
    let overlap = w / 4;
    
    // Create left image (0 to w/2 + overlap)
    let mut left = Mat::new(h, w / 2 + overlap, src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    for row in 0..h {
        for col in 0..(w / 2 + overlap) {
            if col < src.inner.cols() {
                let src_pixel = src.inner.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                let dst_pixel = left.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                for ch in 0..src.inner.channels() {
                    dst_pixel[ch] = src_pixel[ch];
                }
            }
        }
    }
    
    // Create right image (w/2 - overlap to w)
    let mut right = Mat::new(h, w / 2 + overlap, src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    for row in 0..h {
        for col in 0..(w / 2 + overlap) {
            let src_col = (w / 2 - overlap) + col;
            if src_col < src.inner.cols() {
                let src_pixel = src.inner.at(row, src_col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                let dst_pixel = right.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                for ch in 0..src.inner.channels() {
                    dst_pixel[ch] = src_pixel[ch];
                }
            }
        }
    }
    
    // Create masks
    let mut mask_left = Mat::new(h, w / 2 + overlap, 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut mask_right = Mat::new(h, w / 2 + overlap, 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    for row in 0..h {
        for col in 0..(w / 2 + overlap) {
            let left_pix = mask_left.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
            left_pix[0] = 255;
            
            let right_pix = mask_right.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
            right_pix[0] = 255;
        }
    }
    
    // Apply multi-band blending
    let blender = MultiBandBlender::new(num_bands.max(1).min(6));
    let result = blender.blend(&[left, right], &[mask_left, mask_right])
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    Ok(WasmMat { inner: result })
}


// ===== gradientMagnitude =====
#[wasm_bindgen(js_name = gradientMagnitude)]
pub async fn gradient_magnitude_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::gradient_magnitude_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            return Err(JsValue::from_str("CPU not yet implemented for gradient_magnitude"));
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== integralImage =====
#[wasm_bindgen(js_name = integralImage)]
pub async fn integral_image_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::integral_image_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            return Err(JsValue::from_str("CPU not yet implemented for integral_image"));
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== normalize =====
#[wasm_bindgen(js_name = normalize)]
pub async fn normalize_wasm(src: &WasmMat, alpha: f64, beta: f64) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::normalize_gpu_async(&src.inner, &mut dst, alpha, beta)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            return Err(JsValue::from_str("CPU not yet implemented for normalize"));
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== splitChannels =====
#[wasm_bindgen(js_name = splitChannels)]
pub async fn split_channels_wasm(src: &WasmMat) -> Result<Vec<WasmMat>, JsValue> {
    let channels = crate::backend_dispatch! {
        gpu => {
            let mut channels = Vec::new();
            crate::gpu::ops::split_gpu_async(&src.inner, &mut channels)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            channels
        }
        cpu => {
            crate::core::split(&src.inner)
                .map_err(|e| JsValue::from_str(&e.to_string()))?
        }
    };

    Ok(channels.into_iter().map(|mat| WasmMat { inner: mat }).collect())
}


// ===== mergeChannels =====
#[wasm_bindgen(js_name = mergeChannels)]
pub async fn merge_channels_wasm(channels: Vec<WasmMat>) -> Result<WasmMat, JsValue> {
    if channels.is_empty() {
        return Err(JsValue::from_str("At least one channel required"));
    }

    let channel_mats: Vec<Mat> = channels.iter().map(|wm| wm.inner.clone()).collect();
    let rows = channel_mats[0].rows();
    let cols = channel_mats[0].cols();
    let num_channels = channel_mats.len();

    let mut dst = Mat::new(rows, cols, num_channels, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::merge_gpu_async(&channel_mats, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::core::merge(&channel_mats, &mut dst)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


