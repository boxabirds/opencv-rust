//! Image filtering operations

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::Size;
#[cfg(target_arch = "wasm32")]
use crate::wasm::WasmMat;
#[cfg(target_arch = "wasm32")]
use crate::wasm::backend;

/// Gaussian blur operation (WASM-compatible, GPU-accelerated, ASYNC)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = gaussianBlur)]
pub async fn gaussian_blur_wasm(
    src: &WasmMat,
    ksize: usize,
    sigma: f64,
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
                crate::gpu::ops::gaussian_blur_gpu_async(
                    &src.inner,
                    &mut dst,
                    Size::new(ksize as i32, ksize as i32),
                    sigma,
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
            crate::imgproc::gaussian_blur(
                &src.inner,
                &mut dst,
                Size::new(ksize as i32, ksize as i32),
                sigma,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

            Ok(WasmMat { inner: dst })
        }
    }
}

/// Box blur (WASM-compatible)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = blur)]
pub async fn blur_wasm(
    src: &WasmMat,
    ksize: usize,
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
            // GPU path - use box blur as approximation
            #[cfg(feature = "gpu")]
            {
                crate::gpu::ops::box_blur_gpu_async(&src.inner, &mut dst, ksize as i32)
                    .await
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
            crate::imgproc::blur(&src.inner, &mut dst, Size::new(ksize as i32, ksize as i32))
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Box blur - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = boxBlur)]
pub async fn box_blur_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Use backend selection
    match backend::get_backend() {
        1 => {
            // GPU path
            #[cfg(feature = "gpu")]
            {
                crate::gpu::ops::box_blur_gpu_async(&src.inner, &mut dst, ksize)
                    .await
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
            crate::imgproc::blur(&src.inner, &mut dst, Size::new(ksize, ksize))
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Median blur - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = medianBlur)]
pub async fn median_blur_wasm(
    src: &WasmMat,
    ksize: usize,
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
                crate::gpu::ops::median_blur_gpu_async(&src.inner, &mut dst, ksize as i32)
                    .await
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
            crate::imgproc::median_blur(&src.inner, &mut dst, ksize as i32)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Bilateral filter - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = bilateralFilter)]
pub async fn bilateral_filter_wasm(
    src: &WasmMat,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
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
                crate::gpu::ops::bilateral_filter_gpu_async(&src.inner, &mut dst, d, sigma_color, sigma_space)
                    .await
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
            crate::imgproc::bilateral_filter(&src.inner, &mut dst, d, sigma_color, sigma_space)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Apply guided filter for edge-preserving smoothing
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = guidedFilter)]
pub async fn guided_filter_wasm(
    src: &WasmMat,
    radius: i32,
    eps: f64,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::advanced_filter::guided_filter;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale for guide
    let guide = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        src.inner.depth(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Use backend selection (CPU-only for now, future-proof for GPU)
    match backend::get_backend() {
        1 => {
            // GPU path not yet implemented
            #[cfg(feature = "gpu")]
            {
                return Err(JsValue::from_str("GPU guided filter not yet implemented. Try setBackend('cpu')"));
            }
            #[cfg(not(feature = "gpu"))]
            {
                return Err(JsValue::from_str("GPU not available in this build. Try setBackend('cpu')"));
            }
        }
        _ => {
            // CPU path
            guided_filter(&src.inner, &guide, &mut dst, radius, eps)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Apply Gabor filter for texture analysis
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = gaborFilter)]
pub async fn gabor_filter_wasm(
    src: &WasmMat,
    ksize: i32,
    sigma: f64,
    theta: f64,
    lambda: f64,
    gamma: f64,
    psi: f64,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::advanced_filter::gabor_filter;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut dst = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Use backend selection (CPU-only for now, future-proof for GPU)
    match backend::get_backend() {
        1 => {
            // GPU path not yet implemented
            #[cfg(feature = "gpu")]
            {
                return Err(JsValue::from_str("GPU gabor filter not yet implemented. Try setBackend('cpu')"));
            }
            #[cfg(not(feature = "gpu"))]
            {
                return Err(JsValue::from_str("GPU not available in this build. Try setBackend('cpu')"));
            }
        }
        _ => {
            // CPU path
            gabor_filter(&gray, &mut dst, ksize, sigma, theta, lambda, gamma, psi)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== nlmDenoising =====
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

    // Use backend selection (CPU-only for now, future-proof for GPU)
    match backend::get_backend() {
        1 => {
            // GPU path not yet implemented
            #[cfg(feature = "gpu")]
            {
                return Err(JsValue::from_str("GPU NLM denoising not yet implemented. Try setBackend('cpu')"));
            }
            #[cfg(not(feature = "gpu"))]
            {
                return Err(JsValue::from_str("GPU not available in this build. Try setBackend('cpu')"));
            }
        }
        _ => {
            // CPU path
            non_local_means_denoising(&src.inner, &mut dst, h as f32, template_window_size, search_window_size)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== anisotropicDiffusion =====
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

    // Use backend selection (CPU-only for now, future-proof for GPU)
    match backend::get_backend() {
        1 => {
            // GPU path not yet implemented
            #[cfg(feature = "gpu")]
            {
                return Err(JsValue::from_str("GPU anisotropic diffusion not yet implemented. Try setBackend('cpu')"));
            }
            #[cfg(not(feature = "gpu"))]
            {
                return Err(JsValue::from_str("GPU not available in this build. Try setBackend('cpu')"));
            }
        }
        _ => {
            // CPU path
            anisotropic_diffusion(&src.inner, &mut dst, iterations as usize, kappa as f32, lambda as f32)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== fastNlMeans =====
#[wasm_bindgen(js_name = fastNlMeans)]
pub async fn fast_nl_means_wasm(src: &WasmMat, h: f32, template_window_size: i32, search_window_size: i32) -> Result<WasmMat, JsValue> {
    use crate::photo::fast_nl_means_denoising;

    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Use backend selection (CPU-only for now, future-proof for GPU)
    match backend::get_backend() {
        1 => {
            // GPU path not yet implemented
            #[cfg(feature = "gpu")]
            {
                return Err(JsValue::from_str("GPU fast NLM denoising not yet implemented. Try setBackend('cpu')"));
            }
            #[cfg(not(feature = "gpu"))]
            {
                return Err(JsValue::from_str("GPU not available in this build. Try setBackend('cpu')"));
            }
        }
        _ => {
            // CPU path
            fast_nl_means_denoising(&src.inner, &mut dst, h, template_window_size, search_window_size)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== filter2D =====
#[wasm_bindgen(js_name = filter2D)]
pub async fn filter2d_wasm(src: &WasmMat, kernel: Vec<f32>, ksize: usize) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Create Mat from kernel data
    let mut kernel_mat = Mat::new(ksize, ksize, 1, MatDepth::F32)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Convert Vec<f32> to bytes for Mat
    let kernel_bytes: Vec<u8> = kernel.iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();
    kernel_mat.data_mut().copy_from_slice(&kernel_bytes);

    // Anchor point is typically the center of the kernel
    let anchor = ((ksize / 2) as i32, (ksize / 2) as i32);

    // Use backend selection
    match backend::get_backend() {
        1 => {
            // GPU path
            #[cfg(feature = "gpu")]
            {
                crate::gpu::ops::filter2d_gpu_async(&src.inner, &mut dst, &kernel_mat, anchor)
                    .await
                    .map_err(|e| JsValue::from_str(&format!("GPU error: {}. Try setBackend('auto') or setBackend('cpu')", e)))?;
                return Ok(WasmMat { inner: dst });
            }
            #[cfg(not(feature = "gpu"))]
            {
                return Err(JsValue::from_str("GPU not available in this build. Try setBackend('cpu')"));
            }
        }
        _ => {
            // CPU fallback not yet implemented
            return Err(JsValue::from_str("CPU filter2D not yet implemented"));
        }
    }
}


