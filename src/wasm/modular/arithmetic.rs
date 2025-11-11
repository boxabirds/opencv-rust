//! Arithmetic operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::*;
#[cfg(target_arch = "wasm32")]
use crate::wasm::{WasmMat, backend};


/// Element-wise addition - GPU-accelerated
#[cfg(target_arch = "wasm32")]
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


/// Element-wise subtract - GPU-accelerated
#[cfg(target_arch = "wasm32")]
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


/// Element-wise multiply - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = multiply)]
pub async fn multiply_wasm(src1: &WasmMat, src2: &WasmMat, scale: f64) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src1.inner.rows(), src1.inner.cols(), src1.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::multiply_gpu_async(&src1.inner, &src2.inner, &mut dst, scale as f32).await {
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


/// Element-wise minimum - GPU-accelerated
#[cfg(target_arch = "wasm32")]
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


/// Element-wise maximum - GPU-accelerated
#[cfg(target_arch = "wasm32")]
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


/// Element-wise exponential - GPU-accelerated
#[cfg(target_arch = "wasm32")]
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


/// Element-wise logarithm - GPU-accelerated
#[cfg(target_arch = "wasm32")]
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


/// Element-wise square root - GPU-accelerated
#[cfg(target_arch = "wasm32")]
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


/// Element-wise power - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = pow)]
pub async fn pow_wasm(src: &WasmMat, power: f64) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::pow_gpu_async(&src.inner, &mut dst, power as f32).await {
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


/// Add weighted (blend two images) - GPU-accelerated
#[cfg(target_arch = "wasm32")]
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
                alpha as f32,
                &src2.inner,
                beta as f32,
                gamma as f32,
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


/// Convert scale (scale and shift pixel values) - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = convertScale)]
pub async fn convert_scale_wasm(src: &WasmMat, alpha: f64, beta: f64) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::convert_scale_gpu_async(&src.inner, &mut dst, alpha as f32, beta as f32).await {
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


/// Absolute difference - GPU-accelerated
#[cfg(target_arch = "wasm32")]
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


/// Normalize image - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = normalize)]
pub async fn normalize_wasm(src: &WasmMat, alpha: f64, beta: f64) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::normalize_gpu_async(&src.inner, &mut dst, alpha as f32, beta as f32).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU normalize failed, falling back to CPU".into());
                }
            }
        }
    }

    Err(JsValue::from_str("GPU normalize failed and CPU fallback not yet implemented"))
}
