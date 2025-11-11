//! Special operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::*;
#[cfg(target_arch = "wasm32")]
use crate::wasm::{WasmMat, backend};


/// Detect ArUco markers and visualize
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = detectAruco)]
pub async fn detect_aruco_wasm(src: &WasmMat, dict_id: i32) -> Result<WasmMat, JsValue> {
    use crate::objdetect::aruco::{ArucoDetector, ArucoDictionary};
    use crate::imgproc::drawing::{line, circle};
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

    // Map dict_id to ArucoDictionary variant (default to Dict4X4_50)
    let dict = match dict_id {
        0 => ArucoDictionary::Dict4X4_50,
        1 => ArucoDictionary::Dict5X5_50,
        2 => ArucoDictionary::Dict6X6_50,
        _ => ArucoDictionary::Dict4X4_50,
    };
    let detector = ArucoDetector::new(dict);
    let markers = detector.detect_markers(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw markers on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0); // Green

    for marker in markers {
        // Draw marker corners
        for i in 0..4 {
            let p1_f = marker.corners[i];
            let p2_f = marker.corners[(i + 1) % 4];
            let p1 = Point::new(p1_f.x as i32, p1_f.y as i32);
            let p2 = Point::new(p2_f.x as i32, p2_f.y as i32);
            let _ = line(&mut result, p1, p2, color, 2);
            let _ = circle(&mut result, p1, 5, Scalar::new(255.0, 0.0, 0.0, 255.0));
        }
    }

    Ok(WasmMat { inner: result })
}


/// Detect QR codes and visualize
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = detectQR)]
pub async fn detect_qr_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::objdetect::qr_detector::QRCodeDetector;
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

    let detector = QRCodeDetector::new();
    let results = detector.detect_multi(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw QR code boundaries
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 255.0, 255.0); // Magenta

    for qr_points in results {
        if qr_points.len() >= 4 {
            for i in 0..4 {
                let p1_f = qr_points[i];
                let p2_f = qr_points[(i + 1) % 4];
                let p1 = Point::new(p1_f.x as i32, p1_f.y as i32);
                let p2 = Point::new(p2_f.x as i32, p2_f.y as i32);
                let _ = line(&mut result, p1, p2, color, 3);
            }
        }
    }

    Ok(WasmMat { inner: result })
}


/// Laplacian of Gaussian (LoG) blob detection
#[cfg(target_arch = "wasm32")]
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


/// Inpaint - fill missing regions
#[cfg(target_arch = "wasm32")]
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


/// K-means clustering
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = kmeans)]
pub async fn kmeans_wasm(src: &WasmMat, k: usize) -> Result<WasmMat, JsValue> {
    use crate::ml::kmeans::{kmeans, KMeansFlags};

    // Reshape image to points
    let rows = src.inner.rows();
    let cols = src.inner.cols();
    let channels = src.inner.channels();

    let mut points = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let pixel = src.inner.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
            let mut point = Vec::new();
            for ch in 0..channels {
                point.push(pixel[ch] as f64);
            }
            points.push(point);
        }
    }

    // Run k-means
    let mut labels = vec![0i32; points.len()];
    let (centers, _compactness) = kmeans(&points, k, &mut labels, 10, 1.0, KMeansFlags::PPCenters)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Create output image with cluster colors
    let mut result = Mat::new(rows, cols, channels, src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Use cluster centers as colors
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let label = labels[idx] as usize;
            let center = &centers[label % centers.len()];
            let pixel = result.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
            for ch in 0..channels {
                pixel[ch] = center[ch].min(255.0).max(0.0) as u8;
            }
        }
    }

    Ok(WasmMat { inner: result })
}


/// Tonemap Drago for HDR images
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = tonemapDrago)]
pub async fn tonemap_drago_wasm(src: &WasmMat, bias: f64) -> Result<WasmMat, JsValue> {
    use crate::photo::hdr::TonemapDrago;

    let tonemap = TonemapDrago::new().with_bias(bias as f32);
    let dst = tonemap.process(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


/// Tonemap Reinhard for HDR images
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = tonemapReinhard)]
pub async fn tonemap_reinhard_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::photo::hdr::TonemapReinhard;

    let tonemap = TonemapReinhard::new();
    let dst = tonemap.process(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


/// Find homography between matched points
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = findHomography)]
pub async fn find_homography_wasm(src: &WasmMat, n_features: usize) -> Result<WasmMat, JsValue> {
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

    let sift = SIFTF32::new(n_features);
    let (keypoints, _) = sift.detect_and_compute(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Visualize detected keypoints (homography would need two images)
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 255.0, 255.0);

    for kp in keypoints.iter().take(50) {
        let pt = Point::new(kp.pt.x as i32, kp.pt.y as i32);
        let _ = circle(&mut result, pt, 3, color);
    }

    Ok(WasmMat { inner: result })
}


/// Brute force descriptor matcher (simplified - shows keypoint detection)
#[cfg(target_arch = "wasm32")]
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


/// Super resolution
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = superResolution)]
pub async fn super_resolution_wasm(src: &WasmMat, scale: f32) -> Result<WasmMat, JsValue> {
    use crate::photo::super_resolution::SuperResolutionBicubic;

    let sr = SuperResolutionBicubic::new(scale);
    let dst = sr.process(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


/// Merge Debevec (HDR)
#[cfg(target_arch = "wasm32")]
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


/// Gradient magnitude - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = gradientMagnitude)]
pub async fn gradient_magnitude_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::gradient_magnitude_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU gradientMagnitude failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback not yet implemented
    Err(JsValue::from_str("GPU gradientMagnitude failed and CPU fallback not yet implemented"))
}


/// Integral image - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = integralImage)]
pub async fn integral_image_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::integral_image_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU integralImage failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback not yet implemented
    Err(JsValue::from_str("GPU integralImage failed and CPU fallback not yet implemented"))
}


/// Lookup table transform - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = lut)]
pub async fn lut_wasm(src: &WasmMat, table: Vec<u8>) -> Result<WasmMat, JsValue> {
    if table.len() != 256 {
        return Err(JsValue::from_str("LUT table must have exactly 256 entries"));
    }

    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut lut_mat = Mat::new(1, 256, 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    lut_mat.data_mut().copy_from_slice(&table);

    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::lut_gpu_async(&src.inner, &lut_mat, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU LUT failed, falling back to CPU".into());
                }
            }
        }
    }

    Err(JsValue::from_str("GPU LUT failed and CPU fallback not yet implemented"))
}


/// Split multi-channel image into separate channels - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = splitChannels)]
pub async fn split_channels_wasm(src: &WasmMat) -> Result<Vec<WasmMat>, JsValue> {
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::split_gpu_async(&src.inner).await {
                Ok(channels) => {
                    return Ok(channels.into_iter().map(|mat| WasmMat { inner: mat }).collect());
                }
                Err(_) => {
                    web_sys::console::log_1(&"GPU split failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    let channels = crate::core::split(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(channels.into_iter().map(|mat| WasmMat { inner: mat }).collect())
}


/// Merge separate channels into multi-channel image - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = mergeChannels)]
pub async fn merge_channels_wasm(channels: Vec<WasmMat>) -> Result<WasmMat, JsValue> {
    if channels.is_empty() {
        return Err(JsValue::from_str("At least one channel required"));
    }

    let channel_mats: Vec<Mat> = channels.iter().map(|wm| wm.inner.clone()).collect();
    let rows = channel_mats[0].rows();
    let cols = channel_mats[0].cols();
    let num_channels = channel_mats.len() as i32;

    let mut dst = Mat::new(rows, cols, num_channels as usize, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::merge_gpu_async(&channel_mats, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU merge failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::core::merge(&channel_mats, &mut dst)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}
