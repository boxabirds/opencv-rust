//! Stereo operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::*;
#[cfg(target_arch = "wasm32")]
use crate::wasm::{WasmMat, backend};


/// Calibrate camera (simplified demo)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = calibrateCamera)]
pub async fn calibrate_camera_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("calibrate_camera_wasm not yet fully implemented"))
}


#[cfg(feature = "calib_experimental")]
/// Fisheye calibration
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = fisheyeCalibration)]
pub async fn fisheye_calibration_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("fisheye_calibration_wasm not yet fully implemented"))
}


#[cfg(feature = "calib_experimental")]
/// Solve PnP (pose estimation)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = solvePnp)]
pub async fn solve_pnp_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("solve_pnp_wasm not yet fully implemented"))
}


#[cfg(feature = "calib_experimental")]
/// Stereo calibration
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = stereoCalibration)]
pub async fn stereo_calibration_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("stereo_calibration_wasm not yet fully implemented"))
}


/// Compute disparity (stereo matching)
#[cfg(target_arch = "wasm32")]
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


#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = stereoRectification)]
pub async fn stereo_rectification_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("stereo_rectification_wasm not yet fully implemented"))
}
