//! Denoising operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::*;
#[cfg(target_arch = "wasm32")]
use crate::wasm::{WasmMat, backend};


/// Non-Local Means denoising
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = nlmDenoising)]
pub async fn nlm_denoising_wasm(
    src: &WasmMat,
    h: f64,
    template_window_size: i32,
    search_window_size: i32,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::non_local_means_denoising;

    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        src.inner.depth(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    non_local_means_denoising(&src.inner, &mut dst, h as f32, template_window_size, search_window_size)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


/// Fast NL Means denoising
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = fastNlMeans)]
pub async fn fast_nl_means_wasm(src: &WasmMat, h: f32, template_window_size: i32, search_window_size: i32) -> Result<WasmMat, JsValue> {
    use crate::photo::fast_nl_means_denoising;

    let mut dst = Mat::zeros(src.inner.rows(), src.inner.cols(), src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    fast_nl_means_denoising(&src.inner, &mut dst, h, template_window_size, search_window_size)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}


/// Anisotropic diffusion - edge-preserving noise reduction
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = anisotropicDiffusion)]
pub async fn anisotropic_diffusion_wasm(
    src: &WasmMat,
    iterations: i32,
    kappa: f64,
    lambda: f64,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::advanced_filter::anisotropic_diffusion;

    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        src.inner.depth(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    anisotropic_diffusion(&src.inner, &mut dst, iterations as usize, kappa as f32, lambda as f32)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}
