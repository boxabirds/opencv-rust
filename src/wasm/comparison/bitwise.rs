//! WASM bindings

use wasm_bindgen::prelude::*;
use crate::core::{Mat, MatDepth};
use crate::wasm::WasmMat;

// ===== bitwiseNot =====
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


// ===== bitwiseAnd =====
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


// ===== bitwiseOr =====
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


// ===== bitwiseXor =====
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


// ===== absdiff =====
#[wasm_bindgen(js_name = absdiff)]
pub async fn absdiff_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::absdiff_gpu_async(&src1.inner, &src2.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU absdiff failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::core::abs_diff(&src1.inner, &src2.inner, &mut dst)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


// ===== min =====
#[wasm_bindgen(js_name = min)]
pub async fn min_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::min_gpu_async(&src1.inner, &src2.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU min failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback not yet implemented
    Err(JsValue::from_str("GPU min failed and CPU fallback not yet implemented"))
}


// ===== max =====
#[wasm_bindgen(js_name = max)]
pub async fn max_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::max_gpu_async(&src1.inner, &src2.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU max failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback not yet implemented
    Err(JsValue::from_str("GPU max failed and CPU fallback not yet implemented"))
}


// ===== inRange =====
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


// ===== lut =====
#[wasm_bindgen(js_name = lut)]
pub async fn lut_wasm(src: &WasmMat, table: Vec<u8>) -> Result<WasmMat, JsValue> {
    if table.len() != 256 {
        return Err(JsValue::from_str("LUT table must have exactly 256 entries"));
    }

    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut lut_mat = Mat::new(1, 256, 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    lut_mat.data_mut().copy_from_slice(&table);

    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::lut_gpu_async(&src.inner, &mut dst, &lut_mat).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU LUT failed, falling back to CPU".into());
                }
            }
        }
    }

    Err(JsValue::from_str("GPU LUT failed and CPU fallback not yet implemented"))
}


