//! Color conversion WASM bindings

use wasm_bindgen::prelude::*;
use crate::core::{Mat, MatDepth};
use crate::core::types::ColorConversionCode;
use crate::wasm::WasmMat;
use crate::wasm::backend;

/// Convert to grayscale (GPU-accelerated)
#[wasm_bindgen(js_name = cvtColorGray)]
pub async fn cvt_color_gray_wasm(
    src: &WasmMat,
) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let code = if src.inner.channels() == 4 {
        ColorConversionCode::RgbaToGray
    } else {
        ColorConversionCode::RgbToGray
    };

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::rgb_to_gray_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::cvt_color(&src.inner, &mut dst, code)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Convert RGB to HSV color space
#[wasm_bindgen(js_name = cvtColorHsv)]
pub async fn cvt_color_hsv_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::color::cvt_color_async;

    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        src.inner.depth(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::rgb_to_hsv_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::cvt_color(&src.inner, &mut dst, ColorConversionCode::RgbToHsv)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Convert RGB/BGR to Lab color space
#[wasm_bindgen(js_name = cvtColorLab)]
pub async fn cvt_color_lab_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::color::cvt_color;

    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::rgb_to_lab_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            cvt_color(&src.inner, &mut dst, ColorConversionCode::BgrToLab)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Convert RGB/BGR to YCrCb color space
#[wasm_bindgen(js_name = cvtColorYCrCb)]
pub async fn cvt_color_ycrcb_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::color::cvt_color;

    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::rgb_to_ycrcb_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            cvt_color(&src.inner, &mut dst, ColorConversionCode::BgrToYCrCb)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Convert RGB to grayscale (GPU-accelerated)
#[wasm_bindgen(js_name = rgbToGray)]
pub async fn rgb_to_gray_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::rgb_to_gray_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::cvt_color(&src.inner, &mut dst, ColorConversionCode::RgbToGray)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Convert RGB to HSV color space (GPU-accelerated)
#[wasm_bindgen(js_name = rgbToHsv)]
pub async fn rgb_to_hsv_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::rgb_to_hsv_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::cvt_color(&src.inner, &mut dst, ColorConversionCode::RgbToHsv)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Convert RGB to Lab color space (GPU-accelerated)
#[wasm_bindgen(js_name = rgbToLab)]
pub async fn rgb_to_lab_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::rgb_to_lab_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::cvt_color(&src.inner, &mut dst, ColorConversionCode::RgbToLab)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Convert RGB to YCrCb color space (GPU-accelerated)
#[wasm_bindgen(js_name = rgbToYCrCb)]
pub async fn rgb_to_ycrcb_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::rgb_to_ycrcb_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::cvt_color(&src.inner, &mut dst, ColorConversionCode::RgbToYCrCb)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Convert HSV to RGB color space (GPU-accelerated)
#[wasm_bindgen(js_name = hsvToRgb)]
pub async fn hsv_to_rgb_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::hsv_to_rgb_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::cvt_color(&src.inner, &mut dst, ColorConversionCode::HsvToRgb)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Convert Lab to RGB color space (GPU-accelerated)
#[wasm_bindgen(js_name = labToRgb)]
pub async fn lab_to_rgb_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::lab_to_rgb_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::cvt_color(&src.inner, &mut dst, ColorConversionCode::LabToRgb)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Convert YCrCb to RGB color space (GPU-accelerated)
#[wasm_bindgen(js_name = ycrcbToRgb)]
pub async fn ycrcb_to_rgb_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::ycrcb_to_rgb_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::cvt_color(&src.inner, &mut dst, ColorConversionCode::YCrCbToRgb)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}
