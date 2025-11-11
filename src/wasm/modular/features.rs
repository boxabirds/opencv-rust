//! Features operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::*;
#[cfg(target_arch = "wasm32")]
use crate::wasm::{WasmMat, backend};


/// Detect Harris corners and visualize them
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = harrisCorners)]
pub async fn harris_corners_wasm(
    src: &WasmMat,
    block_size: i32,
    ksize: i32,
    k: f64,
    threshold: f64,
) -> Result<WasmMat, JsValue> {
    use crate::features2d::harris_corners;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{ColorConversionCode, Scalar};
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

    // Detect corners
    let keypoints = harris_corners(&gray, block_size, ksize, k, threshold)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0); // Green

    for kp in keypoints {
        circle(&mut result, kp.pt, 3, color)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(WasmMat { inner: result })
}


/// Detect good features to track and visualize them
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = goodFeaturesToTrack)]
pub async fn good_features_to_track_wasm(
    src: &WasmMat,
    max_corners: usize,
    quality_level: f64,
    min_distance: f64,
    block_size: i32,
) -> Result<WasmMat, JsValue> {
    use crate::features2d::good_features_to_track;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{ColorConversionCode, Scalar};
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

    // Detect corners
    let keypoints = good_features_to_track(&gray, max_corners, quality_level, min_distance, block_size)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 0.0, 255.0, 255.0); // Red

    for kp in keypoints {
        circle(&mut result, kp.pt, 5, color)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(WasmMat { inner: result })
}


/// Detect FAST keypoints and visualize them
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = fast)]
pub async fn fast_wasm(
    src: &WasmMat,
    threshold: i32,
    nonmax_suppression: bool,
) -> Result<WasmMat, JsValue> {
    use crate::features2d::fast;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{ColorConversionCode, Scalar};
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

    // Detect keypoints
    let keypoints = fast(&gray, threshold, nonmax_suppression)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 255.0, 0.0, 255.0); // Cyan

    for kp in keypoints {
        circle(&mut result, kp.pt, 2, color)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(WasmMat { inner: result })
}


/// SIFT feature detection and visualization
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = sift)]
pub async fn sift_wasm(src: &WasmMat, n_features: usize) -> Result<WasmMat, JsValue> {
    use crate::features2d::SIFTF32;
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{Point, Scalar};

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

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0);

    for kp in keypoints.iter() {
        let pt = Point::new(kp.pt.x as i32, kp.pt.y as i32);
        let radius = (kp.size / 2.0) as i32;
        let _ = circle(&mut result, pt, radius, color);
    }

    Ok(WasmMat { inner: result })
}


/// ORB feature detection and visualization
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = orb)]
pub async fn orb_wasm(src: &WasmMat, n_features: usize) -> Result<WasmMat, JsValue> {
    use crate::features2d::ORB;
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{Point, Scalar};

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

    let orb = ORB::new(n_features);
    let (keypoints, _) = orb.detect_and_compute(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 0.0, 255.0);

    for kp in keypoints.iter() {
        let pt = Point::new(kp.pt.x as i32, kp.pt.y as i32);
        let radius = (kp.size / 2.0) as i32;
        let _ = circle(&mut result, pt, radius, color);
    }

    Ok(WasmMat { inner: result })
}


/// BRISK feature detection and visualization
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = brisk)]
pub async fn brisk_wasm(src: &WasmMat, threshold: i32) -> Result<WasmMat, JsValue> {
    use crate::features2d::BRISK;
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{Point, Scalar};

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

    let brisk = BRISK::new(threshold, 3);
    let (keypoints, _) = brisk.detect_and_compute(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 255.0, 255.0);

    for kp in keypoints.iter() {
        let pt = Point::new(kp.pt.x as i32, kp.pt.y as i32);
        let radius = (kp.size / 2.0) as i32;
        let _ = circle(&mut result, pt, radius, color);
    }

    Ok(WasmMat { inner: result })
}


/// AKAZE feature detection and visualization
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = akaze)]
pub async fn akaze_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::features2d::AKAZE;
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{Point, Scalar};

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

    let akaze = AKAZE::new();
    let (keypoints, _) = akaze.detect_and_compute(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 255.0, 0.0, 255.0);

    for kp in keypoints.iter() {
        let pt = Point::new(kp.pt.x as i32, kp.pt.y as i32);
        let radius = (kp.size / 2.0) as i32;
        let _ = circle(&mut result, pt, radius, color);
    }

    Ok(WasmMat { inner: result })
}


/// KAZE feature detection and visualization
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = kaze)]
pub async fn kaze_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::features2d::KAZE;
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{Point, Scalar};

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

    let kaze = KAZE::new(false, false);
    let (keypoints, _) = kaze.detect_and_compute(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 255.0, 255.0);

    for kp in keypoints.iter() {
        let pt = Point::new(kp.pt.x as i32, kp.pt.y as i32);
        let radius = (kp.size / 2.0) as i32;
        let _ = circle(&mut result, pt, radius, color);
    }

    Ok(WasmMat { inner: result })
}
