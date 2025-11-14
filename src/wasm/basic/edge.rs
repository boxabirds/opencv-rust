//! Edge detection operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::ColorConversionCode;
#[cfg(target_arch = "wasm32")]
use crate::wasm::WasmMat;
#[cfg(target_arch = "wasm32")]
use crate::wasm::backend;

/// Canny edge detection (WASM-compatible, GPU-accelerated, ASYNC)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = canny)]
pub async fn canny_wasm(
    src: &WasmMat,
    threshold1: f64,
    threshold2: f64,
) -> Result<WasmMat, JsValue> {
    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        // Use correct color conversion based on number of channels
        let conversion_code = if src.inner.channels() == 4 {
            ColorConversionCode::RgbaToGray
        } else {
            ColorConversionCode::BgrToGray
        };
        crate::imgproc::cvt_color(&src.inner, &mut g, conversion_code)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut dst = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::canny_gpu_async(&gray, &mut dst, threshold1, threshold2)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::canny(&gray, &mut dst, threshold1, threshold2)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Sobel edge detection (WASM-compatible)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = sobel)]
pub async fn sobel_wasm(
    src: &WasmMat,
    dx: i32,
    dy: i32,
    ksize: i32,
) -> Result<WasmMat, JsValue> {
    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        // Use correct color conversion based on number of channels
        let conversion_code = if src.inner.channels() == 4 {
            ColorConversionCode::RgbaToGray
        } else {
            ColorConversionCode::BgrToGray
        };
        crate::imgproc::cvt_color(&src.inner, &mut g, conversion_code)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut dst = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::sobel_gpu_async(&gray, &mut dst, dx, dy)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::sobel(&gray, &mut dst, dx, dy, ksize)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Scharr edge detection (WASM-compatible)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = scharr)]
pub async fn scharr_wasm(
    src: &WasmMat,
    dx: i32,
    dy: i32,
) -> Result<WasmMat, JsValue> {
    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        // Use correct color conversion based on number of channels
        let conversion_code = if src.inner.channels() == 4 {
            ColorConversionCode::RgbaToGray
        } else {
            ColorConversionCode::BgrToGray
        };
        crate::imgproc::cvt_color(&src.inner, &mut g, conversion_code)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut dst = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::scharr_gpu_async(&gray, &mut dst, dx, dy)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::scharr(&gray, &mut dst, dx, dy)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Laplacian edge detection (WASM-compatible)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = laplacian)]
pub async fn laplacian_wasm(
    src: &WasmMat,
    ksize: i32,
) -> Result<WasmMat, JsValue> {
    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        // Use correct color conversion based on number of channels
        let conversion_code = if src.inner.channels() == 4 {
            ColorConversionCode::RgbaToGray
        } else {
            ColorConversionCode::BgrToGray
        };
        crate::imgproc::cvt_color(&src.inner, &mut g, conversion_code)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut dst = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::laplacian_gpu_async(&gray, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::laplacian(&gray, &mut dst, ksize)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}
