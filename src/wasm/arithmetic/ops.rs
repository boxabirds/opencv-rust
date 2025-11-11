//! WASM bindings

use wasm_bindgen::prelude::*;
use crate::core::{Mat, MatDepth};
use crate::wasm::WasmMat;

// ===== convertScale =====
#[wasm_bindgen(js_name = convertScale)]
pub async fn convert_scale_wasm(src: &WasmMat, alpha: f64, beta: f64) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::convert_scale_gpu_async(&src.inner, &mut dst, alpha, beta).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU convertScale failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback not yet implemented
    Err(JsValue::from_str("GPU convertScale failed and CPU fallback not yet implemented"))
}


// ===== addWeighted =====
#[wasm_bindgen(js_name = addWeighted)]
pub async fn add_weighted_wasm(src1: &WasmMat, alpha: f64, src2: &WasmMat, beta: f64, gamma: f64) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::add_weighted_gpu_async(
                &src1.inner,
                alpha,
                &src2.inner,
                beta,
                gamma,
                &mut dst
            ).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU addWeighted failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback not yet implemented
    Err(JsValue::from_str("GPU addWeighted failed and CPU fallback not yet implemented"))
}


// ===== add =====
#[wasm_bindgen(js_name = add)]
pub async fn add_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::add_gpu_async(&src1.inner, &src2.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU add failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::core::add(&src1.inner, &src2.inner, &mut dst)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


// ===== pow =====
#[wasm_bindgen(js_name = pow)]
pub async fn pow_wasm(src: &WasmMat, power: f64) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::pow_gpu_async(&src.inner, &mut dst, power).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU pow failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback not yet implemented
    Err(JsValue::from_str("GPU pow failed and CPU fallback not yet implemented"))
}


// ===== subtract =====
#[wasm_bindgen(js_name = subtract)]
pub async fn subtract_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::subtract_gpu_async(&src1.inner, &src2.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU subtract failed, falling back to CPU".into());
                }
            }
        }
    }

    crate::core::subtract(&src1.inner, &src2.inner, &mut dst)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


// ===== multiply =====
#[wasm_bindgen(js_name = multiply)]
pub async fn multiply_wasm(src1: &WasmMat, src2: &WasmMat, scale: f64) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::multiply_gpu_async(&src1.inner, &src2.inner, &mut dst, scale).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU multiply failed, falling back to CPU".into());
                }
            }
        }
    }

    crate::core::multiply(&src1.inner, &src2.inner, &mut dst, scale)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


// ===== exp =====
#[wasm_bindgen(js_name = exp)]
pub async fn exp_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::exp_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU exp failed, falling back to CPU".into());
                }
            }
        }
    }

    Err(JsValue::from_str("GPU exp failed and CPU fallback not yet implemented"))
}


// ===== log =====
#[wasm_bindgen(js_name = log)]
pub async fn log_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::log_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU log failed, falling back to CPU".into());
                }
            }
        }
    }

    Err(JsValue::from_str("GPU log failed and CPU fallback not yet implemented"))
}


// ===== sqrt =====
#[wasm_bindgen(js_name = sqrt)]
pub async fn sqrt_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::sqrt_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU sqrt failed, falling back to CPU".into());
                }
            }
        }
    }

    Err(JsValue::from_str("GPU sqrt failed and CPU fallback not yet implemented"))
}


