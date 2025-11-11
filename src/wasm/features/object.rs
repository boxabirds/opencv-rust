//! WASM bindings

use wasm_bindgen::prelude::*;
use crate::core::{Mat, MatDepth};
use crate::wasm::WasmMat;

// ===== detectAruco =====
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

    let mut result = src.inner.clone();

    crate::backend_dispatch! {
        gpu => {
            return Err(JsValue::from_str("GPU not yet implemented for detect_aruco"));
        }
        cpu => {
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
        }
    }

    Ok(WasmMat { inner: result })
}


// ===== detectQR =====
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

    let mut result = src.inner.clone();

    crate::backend_dispatch! {
        gpu => {
            return Err(JsValue::from_str("GPU not yet implemented for detect_qr"));
        }
        cpu => {
            let detector = QRCodeDetector::new();
            let results = detector.detect_multi(&gray)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Draw QR code boundaries
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
        }
    }

    Ok(WasmMat { inner: result })
}


// ===== hogDescriptor =====
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

    let mut result = src.inner.clone();

    crate::backend_dispatch! {
        gpu => {
            return Err(JsValue::from_str("GPU not yet implemented for hog_descriptor"));
        }
        cpu => {
            let hog = HOGDescriptor::new();
            let _descriptors = hog.compute(&gray)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Draw grid to show HOG cells
            let cell_size = 16;
            let color = Scalar::new(0.0, 255.0, 0.0, 255.0);

            for y in (0..result.rows()).step_by(cell_size) {
                for x in (0..result.cols()).step_by(cell_size) {
                    let rect = Rect::new(x as i32, y as i32, cell_size as i32, cell_size as i32);
                    let _ = rectangle(&mut result, rect, color, 1);
                }
            }
        }
    }

    Ok(WasmMat { inner: result })
}


// ===== cascadeClassifier =====
#[wasm_bindgen(js_name = cascadeClassifier)]
pub async fn cascade_classifier_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::objdetect::cascade::CascadeClassifier;
    use crate::imgproc::drawing::rectangle;
    use crate::imgproc::color::cvt_color;
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

    let mut result = src.inner.clone();

    crate::backend_dispatch! {
        gpu => {
            return Err(JsValue::from_str("GPU not yet implemented for cascade_classifier"));
        }
        cpu => {
            // Create classifier (would need cascade file in production)
            let cascade = CascadeClassifier::new();
            let detections = cascade.detect_multi_scale(&gray, 1.1, 3, (30, 30), (100, 100))
                .unwrap_or_else(|_| vec![]);

            // Draw detections
            let color = Scalar::new(0.0, 255.0, 0.0, 255.0);

            for rect in detections.iter().take(10) {
                let _ = rectangle(&mut result, *rect, color, 2);
            }
        }
    }

    Ok(WasmMat { inner: result })
}


