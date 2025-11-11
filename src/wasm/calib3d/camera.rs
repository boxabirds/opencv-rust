//! WASM bindings

use wasm_bindgen::prelude::*;
use crate::core::{Mat, MatDepth};
use crate::wasm::WasmMat;

// ===== findHomography =====
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


// ===== calibrateCamera =====
#[wasm_bindgen(js_name = calibrateCamera)]
pub async fn calibrate_camera_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::put_text;
    use crate::core::types::{Point, Scalar};

    // Simplified demo - would need checkerboard pattern in production
    let mut result = src.inner.clone();
    let text = "Camera calibration demo".to_string();
    let _ = put_text(&mut result, &text, Point::new(10, 30), 0.7, Scalar::new(255.0, 255.0, 255.0, 255.0));

    Ok(WasmMat { inner: result })
}


// ===== fisheyeCalibration =====
#[wasm_bindgen(js_name = fisheyeCalibration)]
pub async fn fisheye_calibration_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::put_text;
    use crate::core::types::{Point, Scalar};

    // Simplified demo
    let mut result = src.inner.clone();
    let text = "Fisheye calibration demo".to_string();
    let _ = put_text(&mut result, &text, Point::new(10, 30), 0.7, Scalar::new(0.0, 255.0, 255.0, 255.0));

    Ok(WasmMat { inner: result })
}


// ===== solvePnp =====
#[wasm_bindgen(js_name = solvePnp)]
pub async fn solve_pnp_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::{put_text, circle};
    use crate::core::types::{Point, Scalar};

    // Simplified demo - show reference points
    let mut result = src.inner.clone();
    let text = "PnP pose estimation".to_string();
    let _ = put_text(&mut result, &text, Point::new(10, 30), 0.7, Scalar::new(255.0, 0.0, 255.0, 255.0));

    // Draw some reference points
    let points = vec![
        Point::new(result.cols() as i32 / 4, result.rows() as i32 / 4),
        Point::new(3 * result.cols() as i32 / 4, result.rows() as i32 / 4),
        Point::new(result.cols() as i32 / 2, result.rows() as i32 / 2),
    ];
    for pt in points {
        let _ = circle(&mut result, pt, 5, Scalar::new(255.0, 0.0, 0.0, 255.0));
    }

    Ok(WasmMat { inner: result })
}


// ===== stereoCalibration =====
#[wasm_bindgen(js_name = stereoCalibration)]
pub async fn stereo_calibration_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::put_text;
    use crate::core::types::{Point, Scalar};

    // Simplified demo
    let mut result = src.inner.clone();
    let text = "Stereo calibration demo".to_string();
    let _ = put_text(&mut result, &text, Point::new(10, 30), 0.7, Scalar::new(128.0, 255.0, 128.0, 255.0));

    Ok(WasmMat { inner: result })
}


// ===== computeDisparity =====
#[wasm_bindgen(js_name = computeDisparity)]
pub async fn compute_disparity_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;

    // Simplified: Use shifted image as "right" view for demo
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    // Create shifted version as disparity map demo
    let mut disparity = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    for row in 0..gray.rows() {
        for col in 0..gray.cols() {
            let shift = (col % 20) as u8;
            disparity.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] = shift * 12;
        }
    }

    Ok(WasmMat { inner: disparity })
}


// ===== stereoRectification =====
#[wasm_bindgen(js_name = stereoRectification)]
pub async fn stereo_rectification_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::calib3d::stereo::{stereo_rectify, StereoParameters};
    use crate::calib3d::camera::{CameraMatrix, DistortionCoefficients};
    
    // Create simplified stereo rectification demo
    // Use dummy camera matrices for visualization
    let fx = 500.0;
    let fy = 500.0;
    let cx = (src.inner.cols() / 2) as f64;
    let cy = (src.inner.rows() / 2) as f64;
    
    let camera_left = CameraMatrix {
        fx, fy,
        cx, cy,
    };

    let camera_right = CameraMatrix {
        fx, fy,
        cx: cx + 50.0, // Slight offset for stereo
        cy,
    };

    let dist_left = DistortionCoefficients {
        k: [0.0, 0.0, 0.0],
        p: [0.0, 0.0],
    };
    
    let dist_right = dist_left.clone();

    let rotation = [[1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]];

    let translation = [100.0, 0.0, 0.0]; // 100mm baseline

    let image_size = (src.inner.cols(), src.inner.rows());

    let stereo_params = StereoParameters {
        camera_matrix_left: camera_left,
        camera_matrix_right: camera_right,
        dist_coeffs_left: dist_left,
        dist_coeffs_right: dist_right,
        rotation,
        translation,
        essential_matrix: [[0.0; 3]; 3],
        fundamental_matrix: [[0.0; 3]; 3],
    };

    let (r1, r2, p1, p2) = stereo_rectify(
        &stereo_params,
        image_size,
    ).map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    // Draw grid overlay to show rectification effect
    use crate::imgproc::drawing::line;
    use crate::core::types::{Point, Scalar};
    
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0);
    
    // Draw horizontal lines to show epipolar lines are aligned
    for y in (0..result.rows()).step_by(result.rows() / 10) {
        let pt1 = Point::new(0, y as i32);
        let pt2 = Point::new(result.cols() as i32, y as i32);
        let _ = line(&mut result, pt1, pt2, color, 1);
    }
    
    Ok(WasmMat { inner: result })
}


