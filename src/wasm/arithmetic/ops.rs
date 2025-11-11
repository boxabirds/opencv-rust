//! WASM bindings

use wasm_bindgen::prelude::*;
use crate::core::{Mat, MatDepth};
use crate::wasm::WasmMat;

// ===== convertScale =====
#[wasm_bindgen(js_name = convertScale)]
pub async fn convert_scale_wasm(src: &WasmMat, alpha: f64, beta: f64) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::convert_scale_gpu_async(&src.inner, &mut dst, alpha, beta)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::core::convert_scale_abs(&src.inner, &mut dst, alpha, beta)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== addWeighted =====
#[wasm_bindgen(js_name = addWeighted)]
pub async fn add_weighted_wasm(src1: &WasmMat, alpha: f64, src2: &WasmMat, beta: f64, gamma: f64) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::add_weighted_gpu_async(
                &src1.inner,
                alpha,
                &src2.inner,
                beta,
                gamma,
                &mut dst
            )
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::core::add_weighted(&src1.inner, alpha, &src2.inner, beta, gamma, &mut dst)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== add =====
#[wasm_bindgen(js_name = add)]
pub async fn add_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::add_gpu_async(&src1.inner, &src2.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::core::add(&src1.inner, &src2.inner, &mut dst)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== pow =====
#[wasm_bindgen(js_name = pow)]
pub async fn pow_wasm(src: &WasmMat, power: f64) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::pow_gpu_async(&src.inner, &mut dst, power)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::core::pow(&src.inner, power, &mut dst)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== subtract =====
#[wasm_bindgen(js_name = subtract)]
pub async fn subtract_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::subtract_gpu_async(&src1.inner, &src2.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::core::subtract(&src1.inner, &src2.inner, &mut dst)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== multiply =====
#[wasm_bindgen(js_name = multiply)]
pub async fn multiply_wasm(src1: &WasmMat, src2: &WasmMat, scale: f64) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::multiply_gpu_async(&src1.inner, &src2.inner, &mut dst, scale)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::core::multiply(&src1.inner, &src2.inner, &mut dst, scale)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== exp =====
#[wasm_bindgen(js_name = exp)]
pub async fn exp_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::exp_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::core::exp(&src.inner, &mut dst)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== log =====
#[wasm_bindgen(js_name = log)]
pub async fn log_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::log_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::core::log(&src.inner, &mut dst)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== sqrt =====
#[wasm_bindgen(js_name = sqrt)]
pub async fn sqrt_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::sqrt_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::core::sqrt(&src.inner, &mut dst)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


