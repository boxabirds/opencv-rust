//! WASM bindings

use wasm_bindgen::prelude::*;
use crate::core::{Mat, MatDepth};
use crate::wasm::WasmMat;

// ===== bitwiseNot =====
#[wasm_bindgen(js_name = bitwiseNot)]
pub async fn bitwise_not_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::bitwise_not_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::core::bitwise_not(&src.inner, &mut dst)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== bitwiseAnd =====
#[wasm_bindgen(js_name = bitwiseAnd)]
pub async fn bitwise_and_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::bitwise_and_gpu_async(&src1.inner, &src2.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::core::bitwise_and(&src1.inner, &src2.inner, &mut dst)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== bitwiseOr =====
#[wasm_bindgen(js_name = bitwiseOr)]
pub async fn bitwise_or_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::bitwise_or_gpu_async(&src1.inner, &src2.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::core::bitwise_or(&src1.inner, &src2.inner, &mut dst)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== bitwiseXor =====
#[wasm_bindgen(js_name = bitwiseXor)]
pub async fn bitwise_xor_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::bitwise_xor_gpu_async(&src1.inner, &src2.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            return Err(JsValue::from_str("CPU not yet implemented for bitwise_xor"));
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== absdiff =====
#[wasm_bindgen(js_name = absdiff)]
pub async fn absdiff_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::absdiff_gpu_async(&src1.inner, &src2.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::core::abs_diff(&src1.inner, &src2.inner, &mut dst)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== min =====
#[wasm_bindgen(js_name = min)]
pub async fn min_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::min_gpu_async(&src1.inner, &src2.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            return Err(JsValue::from_str("CPU not yet implemented for min"));
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== max =====
#[wasm_bindgen(js_name = max)]
pub async fn max_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::max_gpu_async(&src1.inner, &src2.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            return Err(JsValue::from_str("CPU not yet implemented for max"));
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== inRange =====
#[wasm_bindgen(js_name = inRange)]
pub async fn in_range_wasm(src: &WasmMat, lower_r: u8, lower_g: u8, lower_b: u8, upper_r: u8, upper_g: u8, upper_b: u8) -> Result<WasmMat, JsValue> {
    use crate::core::types::Scalar;

    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let lower_scalar = Scalar::from_rgb(lower_r, lower_g, lower_b);
    let upper_scalar = Scalar::from_rgb(upper_r, upper_g, upper_b);

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::in_range_gpu_async(&src.inner, &mut dst, lower_scalar, upper_scalar)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            return Err(JsValue::from_str("CPU not yet implemented for in_range"));
        }
    }

    Ok(WasmMat { inner: dst })
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

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::lut_gpu_async(&src.inner, &mut dst, &lut_mat)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            return Err(JsValue::from_str("CPU not yet implemented for lut"));
        }
    }

    Ok(WasmMat { inner: dst })
}


