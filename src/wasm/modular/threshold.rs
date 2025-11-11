//! Threshold operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::*;
#[cfg(target_arch = "wasm32")]
use crate::wasm::{WasmMat, backend};


/// Threshold operation (WASM-compatible, GPU-accelerated, ASYNC)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = threshold)]
pub async fn threshold_wasm(
    src: &WasmMat,
    thresh: f64,
    max_val: f64,
) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        MatDepth::U8,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Use backend selection
    match backend::get_backend() {
        1 => {
            // GPU path
            #[cfg(feature = "gpu")]
            {
                crate::gpu::ops::threshold_gpu_async(
                    &src.inner,
                    &mut dst,
                    thresh as u8,
                    max_val as u8,
                ).await
                .map_err(|e| JsValue::from_str(&format!("GPU error: {}. Try setBackend('auto') or setBackend('cpu')", e)))?;

                return Ok(WasmMat { inner: dst });
            }

            #[cfg(not(feature = "gpu"))]
            {
                return Err(JsValue::from_str("GPU not available in this build. Try setBackend('cpu')"));
            }
        }
        _ => {
            // CPU path
            // Convert to grayscale if needed
            let gray = if src.inner.channels() > 1 {
                let mut gray = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
                    .map_err(|e| JsValue::from_str(&e.to_string()))?;
                crate::imgproc::cvt_color(
                    &src.inner,
                    &mut gray,
                    ColorConversionCode::BgrToGray,
                )
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
                gray
            } else {
                src.inner.clone()
            };

            dst = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            crate::imgproc::threshold(
                &gray,
                &mut dst,
                thresh,
                max_val,
                ThresholdType::Binary,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

            Ok(WasmMat { inner: dst })
        }
    }
}


/// Adaptive threshold (WASM-compatible)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = adaptiveThreshold)]
pub async fn adaptive_threshold_wasm(
    src: &WasmMat,
    maxval: f64,
    block_size: i32,
    c: f64,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::threshold::AdaptiveThresholdMethod;
    use crate::core::types::ThresholdType;

    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        crate::imgproc::cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut dst = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::imgproc::adaptive_threshold_async(
        &gray,
        &mut dst,
        maxval,
        AdaptiveThresholdMethod::Mean,
        ThresholdType::Binary,
        block_size,
        c,
        true, // use_gpu=true for WASM
    )
    .await
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


/// In range (color/value range masking) - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = inRange)]
pub async fn in_range_wasm(src: &WasmMat, lower_r: u8, lower_g: u8, lower_b: u8, upper_r: u8, upper_g: u8, upper_b: u8) -> Result<WasmMat, JsValue> {
    use crate::core::types::Scalar;

    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let lower_scalar = Scalar::from_rgb(lower_r, lower_g, lower_b);
    let upper_scalar = Scalar::from_rgb(upper_r, upper_g, upper_b);
    let lower = [lower_r, lower_g, lower_b];
    let upper = [upper_r, upper_g, upper_b];

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::in_range_gpu_async(&src.inner, &mut dst, lower_scalar, upper_scalar).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU inRange failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback not yet implemented
    Err(JsValue::from_str("GPU inRange failed and CPU fallback not yet implemented"))
}
