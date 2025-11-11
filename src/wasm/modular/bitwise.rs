//! Bitwise operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::*;
#[cfg(target_arch = "wasm32")]
use crate::wasm::{WasmMat, backend};


/// Bitwise NOT - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = bitwiseNot)]
pub async fn bitwise_not_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::bitwise_not_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU bitwiseNot failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::core::bitwise_not(&src.inner, &mut dst)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


/// Bitwise AND - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = bitwiseAnd)]
pub async fn bitwise_and_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::bitwise_and_gpu_async(&src1.inner, &src2.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU bitwiseAnd failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::core::bitwise_and(&src1.inner, &src2.inner, &mut dst)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


/// Bitwise OR - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = bitwiseOr)]
pub async fn bitwise_or_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::bitwise_or_gpu_async(&src1.inner, &src2.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU bitwiseOr failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::core::bitwise_or(&src1.inner, &src2.inner, &mut dst)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


/// Bitwise XOR - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = bitwiseXor)]
pub async fn bitwise_xor_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::bitwise_xor_gpu_async(&src1.inner, &src2.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU bitwiseXor failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback not yet implemented
    Err(JsValue::from_str("GPU bitwiseXor failed and CPU fallback not yet implemented"))
}
