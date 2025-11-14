//! WASM bindings

use wasm_bindgen::prelude::*;
use crate::core::{Mat, MatDepth};
use crate::wasm::WasmMat;

// ===== watershed =====
#[wasm_bindgen(js_name = watershed)]
pub async fn watershed_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::advanced_filter::watershed;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::threshold::threshold;
    use crate::core::types::{ColorConversionCode, ThresholdType};

    // Ensure 3-channel image for watershed
    let bgr = if src.inner.channels() == 1 {
        let mut color = Mat::new(src.inner.rows(), src.inner.cols(), 3, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut color, ColorConversionCode::GrayToBgr)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        color
    } else {
        src.inner.clone()
    };

    // Create markers using simple threshold-based initialization
    let gray = if bgr.channels() > 1 {
        let mut g = Mat::new(bgr.rows(), bgr.cols(), 1, bgr.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        // Use correct color conversion based on number of channels
        let conversion_code = if bgr.channels() == 4 {
            ColorConversionCode::RgbaToGray
        } else {
            ColorConversionCode::BgrToGray
        };
        cvt_color(&bgr, &mut g, conversion_code)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        bgr.clone()
    };

    let mut markers = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Initialize markers: foreground (label 1), background (label 2), unknown (0)
    for row in 0..markers.rows() {
        for col in 0..markers.cols() {
            let val = gray.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0];
            markers.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] = if val < 50 {
                1 // Foreground
            } else if val > 200 {
                2 // Background
            } else {
                0 // Unknown
            };
        }
    }

    watershed(&bgr, &mut markers)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Visualize markers - multiply by 50 to make labels visible
    let mut result = Mat::new(markers.rows(), markers.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    for row in 0..markers.rows() {
        for col in 0..markers.cols() {
            let marker = markers.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0];
            result.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] = (marker.saturating_mul(50)).min(255);
        }
    }

    Ok(WasmMat { inner: result })
}


// ===== kmeans =====
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
                pixel[ch] = center[ch].clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(WasmMat { inner: result })
}


