//! Tracking operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::*;
#[cfg(target_arch = "wasm32")]
use crate::wasm::{WasmMat, backend};


/// Farneback dense optical flow
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = farnebackOpticalFlow)]
pub async fn farneback_optical_flow_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::video::optical_flow::calc_optical_flow_farneback;
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;

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

    // Create a shifted version as "next frame"
    let mut next_frame = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    for row in 0..gray.rows() {
        for col in 5..gray.cols() {
            next_frame.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] = 
                gray.at(row, col - 5).map_err(|e| JsValue::from_str(&e.to_string()))?[0];
        }
    }

    let flow = calc_optical_flow_farneback(&gray, &next_frame, 0.5, 3, 15, 3)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Visualize flow as grayscale magnitude
    let mut result = Mat::new(flow.rows(), flow.cols(), 1, src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    for row in 0..flow.rows() {
        for col in 0..flow.cols() {
            let fx = flow.at_f32(row, col, 0).unwrap_or(0.0);
            let fy = flow.at_f32(row, col, 1).unwrap_or(0.0);
            let mag = (fx * fx + fy * fy).sqrt();
            result.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] = 
                (mag * 10.0).min(255.0) as u8;
        }
    }

    Ok(WasmMat { inner: result })
}


/// MeanShift object tracking
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = meanshiftTracker)]
pub async fn meanshift_tracker_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::video::tracking::MeanShiftTracker;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{Rect, Scalar};

    // Initialize tracker with center region
    let w = src.inner.cols() as i32;
    let h = src.inner.rows() as i32;
    let initial_window = Rect::new(w / 4, h / 4, w / 2, h / 2);

    let mut tracker = MeanShiftTracker::new(initial_window);
    let result_window = tracker.track(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw tracked region
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0);
    let _ = rectangle(&mut result, result_window, color, 2);

    Ok(WasmMat { inner: result })
}


/// CAMShift tracking
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = camshiftTracker)]
pub async fn camshift_tracker_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::video::tracking::CamShiftTracker;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{Rect, Scalar};

    // Initialize tracker with center region
    let w = src.inner.cols() as i32;
    let h = src.inner.rows() as i32;
    let initial_window = Rect::new(w / 4, h / 4, w / 2, h / 2);

    let mut tracker = CamShiftTracker::new(initial_window);
    let result_window = tracker.track(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw tracked region
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 0.0, 255.0);
    let _ = rectangle(&mut result, result_window, color, 2);

    Ok(WasmMat { inner: result })
}


/// MOSSE tracker
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = mosseTracker)]
pub async fn mosse_tracker_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::video::advanced_tracking::MOSSETracker;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{Rect, Scalar};

    // Initialize tracker with center region
    let w = src.inner.cols() as i32;
    let h = src.inner.rows() as i32;
    let initial_bbox = Rect::new(w / 4, h / 4, w / 2, h / 2);

    let mut tracker = MOSSETracker::new();
    tracker.init(&src.inner, initial_bbox)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    let result_bbox = tracker.update(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw tracked region
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 255.0, 0.0, 255.0);
    let _ = rectangle(&mut result, result_bbox, color, 2);

    Ok(WasmMat { inner: result })
}


/// CSRT tracker
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = csrtTracker)]
pub async fn csrt_tracker_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::video::advanced_tracking::CSRTTracker;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{Rect, Scalar};

    // Initialize tracker with center region
    let w = src.inner.cols() as i32;
    let h = src.inner.rows() as i32;
    let initial_bbox = Rect::new(w / 4, h / 4, w / 2, h / 2);

    let mut tracker = CSRTTracker::new();
    tracker.init(&src.inner, initial_bbox)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    let result_bbox = tracker.update(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw tracked region
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 255.0, 255.0);
    let _ = rectangle(&mut result, result_bbox, color, 2);

    Ok(WasmMat { inner: result })
}


/// Background subtractor MOG2
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = bgSubtractorMog2)]
pub async fn bg_subtractor_mog2_wasm(src: &WasmMat, learning_rate: f64) -> Result<WasmMat, JsValue> {
    use crate::video::background_subtraction::BackgroundSubtractorMOG2;

    let mut bg_sub = BackgroundSubtractorMOG2::new();
    let mut fg_mask = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    bg_sub.apply(&src.inner, &mut fg_mask, learning_rate)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: fg_mask })
}


/// Background subtractor KNN
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = bgSubtractorKnn)]
pub async fn bg_subtractor_knn_wasm(src: &WasmMat, learning_rate: f64) -> Result<WasmMat, JsValue> {
    use crate::video::background_subtraction::BackgroundSubtractorKNN;

    let mut bg_sub = BackgroundSubtractorKNN::new();
    let mut fg_mask = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    bg_sub.apply(&src.inner, &mut fg_mask, learning_rate)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: fg_mask })
}


/// HOG (Histogram of Oriented Gradients) descriptor
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = hogDescriptor)]
pub async fn hog_descriptor_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::objdetect::hog::HOGDescriptor;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{ColorConversionCode, Rect, Scalar};

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

    let hog = HOGDescriptor::new();
    let _descriptors = hog.compute(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw grid to show HOG cells
    let mut result = src.inner.clone();
    let cell_size = 16;
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0);

    for y in (0..result.rows()).step_by(cell_size) {
        for x in (0..result.cols()).step_by(cell_size) {
            let rect = Rect::new(x as i32, y as i32, cell_size as i32, cell_size as i32);
            let _ = rectangle(&mut result, rect, color, 1);
        }
    }

    Ok(WasmMat { inner: result })
}
