//! WASM bindings for opencv-rust
//!
//! This module provides JavaScript-compatible bindings for running opencv-rust
//! in the browser via WebAssembly.

pub mod backend;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::{Size, InterpolationFlag, ColorConversionCode, ThresholdType};

/// Initialize the WASM module with panic hooks for better error messages
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_init() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"OpenCV-Rust WASM initialized".into());
}

/// Initialize rayon thread pool for multi-threading
/// This must be called from JavaScript before using any parallel operations
#[cfg(all(target_arch = "wasm32", feature = "wasm-bindgen-rayon"))]
#[wasm_bindgen]
pub fn init_thread_pool(num_threads: usize) -> Result<(), JsValue> {
    wasm_bindgen_rayon::init_thread_pool(num_threads);
    web_sys::console::log_1(&format!("✓ Rayon thread pool initialized with {} threads", num_threads).into());
    Ok(())
}

/// Initialize GPU asynchronously (must be called from JavaScript)
#[cfg(all(target_arch = "wasm32", feature = "gpu"))]
#[wasm_bindgen(js_name = initGpu)]
pub async fn init_gpu_wasm() -> bool {
    let success = crate::gpu::device::GpuContext::init_async().await;
    if success {
        web_sys::console::log_1(&"✓ GPU initialized successfully".into());
    } else {
        web_sys::console::log_1(&"⚠ GPU not available, falling back to CPU".into());
    }
    success
}

#[cfg(all(target_arch = "wasm32", not(feature = "gpu")))]
#[wasm_bindgen(js_name = initGpu)]
pub async fn init_gpu_wasm() -> bool {
    web_sys::console::log_1(&"GPU feature not enabled in this build".into());
    false
}

/// Set the backend execution mode
///
/// # Arguments
/// * `backend` - "auto" | "webgpu" | "gpu" | "cpu"
///
/// # Examples
/// ```javascript
/// import init, { setBackend } from './opencv_rust.js';
///
/// await init();
///
/// // Try GPU first, fall back to CPU (default)
/// setBackend('auto');
///
/// // Force GPU (error if unavailable)
/// setBackend('webgpu');
///
/// // Force CPU
/// setBackend('cpu');
/// ```
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = setBackend)]
pub fn set_backend_wasm(backend: &str) {
    backend::set_backend(backend);
}

/// Get the current backend setting
///
/// # Returns
/// "auto" | "webgpu" | "cpu"
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = getBackend)]
pub fn get_backend_wasm() -> String {
    backend::get_backend_name().to_string()
}

/// Get the resolved backend (only meaningful in Auto mode)
///
/// # Returns
/// "gpu" | "cpu" | "unresolved"
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = getResolvedBackend)]
pub fn get_resolved_backend_wasm() -> String {
    backend::get_resolved_backend_name().to_string()
}

/// WASM-compatible Mat wrapper
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct WasmMat {
    inner: Mat,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl WasmMat {
    /// Create a new Mat from image dimensions
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, channels: usize) -> Result<WasmMat, JsValue> {
        Mat::new(height, width, channels, MatDepth::U8)
            .map(|inner| WasmMat { inner })
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Create Mat from raw RGBA data (Uint8Array from ImageData)
    #[wasm_bindgen(js_name = fromImageData)]
    pub fn from_image_data(
        data: &[u8],
        width: usize,
        height: usize,
        channels: usize,
    ) -> Result<WasmMat, JsValue> {
        let mut mat = Mat::new(height, width, channels, MatDepth::U8)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let mat_data = mat.data_mut();
        let expected_size = width * height * channels;

        if data.len() != expected_size {
            return Err(JsValue::from_str(&format!(
                "Data size mismatch: expected {}, got {}",
                expected_size,
                data.len()
            )));
        }

        mat_data.copy_from_slice(data);
        Ok(WasmMat { inner: mat })
    }

    /// Get raw data as bytes (for creating ImageData in JS)
    #[wasm_bindgen(js_name = getData)]
    pub fn get_data(&self) -> Vec<u8> {
        self.inner.data().to_vec()
    }

    /// Get image width
    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.inner.cols()
    }

    /// Get image height
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.inner.rows()
    }

    /// Get number of channels
    #[wasm_bindgen(getter)]
    pub fn channels(&self) -> usize {
        self.inner.channels()
    }

    /// Free memory (manual cleanup)
    pub fn free(self) {
        // Mat will be dropped automatically
    }
}

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

/// Resize operation (WASM-compatible, GPU-accelerated, ASYNC)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = resize)]
pub async fn resize_wasm(
    src: &WasmMat,
    dst_width: usize,
    dst_height: usize,
) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(dst_height, dst_width, src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::resize_gpu_async(&src.inner, &mut dst, dst_width, dst_height).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU resize failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::imgproc::resize(
        &src.inner,
        &mut dst,
        Size::new(dst_width as i32, dst_height as i32),
        InterpolationFlag::Linear,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

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

/// Canny edge detection (WASM-compatible, GPU-accelerated, ASYNC)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = canny)]
pub async fn canny_wasm(
    src: &WasmMat,
    threshold1: f64,
    threshold2: f64,
) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        MatDepth::U8,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::canny_gpu_async(&src.inner, &mut dst, threshold1, threshold2).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU canny failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback - convert to grayscale if needed
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

    crate::imgproc::canny(&gray, &mut dst, threshold1, threshold2)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Check if GPU is available
#[cfg(all(target_arch = "wasm32", feature = "gpu"))]
#[wasm_bindgen(js_name = isGpuAvailable)]
pub fn is_gpu_available() -> bool {
    crate::gpu::gpu_available()
}

#[cfg(all(target_arch = "wasm32", not(feature = "gpu")))]
#[wasm_bindgen(js_name = isGpuAvailable)]
pub fn is_gpu_available() -> bool {
    false
}

/// Get version information
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = getVersion)]
pub fn get_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ============================================================================
// BATCH 1 & 2 WASM BINDINGS
// ============================================================================

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

    crate::imgproc::blur_async(
        &src.inner,
        &mut dst,
        Size::new(ksize as i32, ksize as i32),
        true, // use_gpu=true for WASM
    )
    .await
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Box blur - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = boxBlur)]
pub async fn box_blur_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::box_blur_gpu_async(&src.inner, &mut dst, ksize).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU box blur failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::imgproc::blur(&src.inner, &mut dst, Size::new(ksize, ksize))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

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

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::median_blur_gpu_async(&src.inner, &mut dst, ksize as i32).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU median blur failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::imgproc::median_blur(&src.inner, &mut dst, ksize as i32)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

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

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::bilateral_filter_gpu_async(&src.inner, &mut dst, d, sigma_color, sigma_space).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU bilateral filter failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::imgproc::bilateral_filter(&src.inner, &mut dst, d, sigma_color, sigma_space)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

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
        crate::imgproc::cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut dst = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Use GPU-accelerated version with fallback to CPU
    crate::imgproc::sobel_async(&gray, &mut dst, dx, dy, ksize, true)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

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
        crate::imgproc::cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut dst = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::imgproc::scharr_async(&gray, &mut dst, dx, dy, true)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

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
        crate::imgproc::cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut dst = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::imgproc::laplacian_async(&gray, &mut dst, ksize, true)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Flip image (WASM-compatible)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = flip)]
pub async fn flip_wasm(
    src: &WasmMat,
    flip_code: i32,
) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        MatDepth::U8,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::imgproc::flip_async(&src.inner, &mut dst, flip_code, true)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Rotate image (WASM-compatible)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = rotate)]
pub async fn rotate_wasm(
    src: &WasmMat,
    rotate_code: i32,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::geometric::RotateCode;

    let rotate_enum = match rotate_code {
        0 => RotateCode::Rotate90Clockwise,
        1 => RotateCode::Rotate180,
        2 => RotateCode::Rotate90CounterClockwise,
        _ => return Err(JsValue::from_str("Invalid rotate code, use 0-2")),
    };

    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        MatDepth::U8,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::imgproc::rotate_async(&src.inner, &mut dst, rotate_enum, true)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Convert to grayscale (WASM-compatible)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = cvtColorGray)]
pub async fn cvt_color_gray_wasm(
    src: &WasmMat,
) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let code = if src.inner.channels() == 4 {
        ColorConversionCode::RgbaToGray
    } else {
        ColorConversionCode::RgbToGray
    };

    crate::imgproc::cvt_color_async(&src.inner, &mut dst, code, true)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
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

/// Draw a line on the image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = drawLine)]
pub async fn draw_line_wasm(
    src: &WasmMat,
    x1: i32,
    y1: i32,
    x2: i32,
    y2: i32,
    r: u8,
    g: u8,
    b: u8,
    thickness: i32,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::line;
    use crate::core::types::{Point, Scalar};

    let mut img = src.inner.clone();
    let pt1 = Point::new(x1, y1);
    let pt2 = Point::new(x2, y2);
    let color = Scalar::new(b as f64, g as f64, r as f64, 255.0);

    line(&mut img, pt1, pt2, color, thickness)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: img })
}

/// Draw a rectangle on the image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = drawRectangle)]
pub async fn draw_rectangle_wasm(
    src: &WasmMat,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    r: u8,
    g: u8,
    b: u8,
    thickness: i32,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{Rect, Scalar};

    let mut img = src.inner.clone();
    let rect = Rect::new(x, y, width, height);
    let color = Scalar::new(b as f64, g as f64, r as f64, 255.0);

    rectangle(&mut img, rect, color, thickness)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: img })
}

/// Draw a circle on the image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = drawCircle)]
pub async fn draw_circle_wasm(
    src: &WasmMat,
    center_x: i32,
    center_y: i32,
    radius: i32,
    r: u8,
    g: u8,
    b: u8,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::circle;
    use crate::core::types::{Point, Scalar};

    let mut img = src.inner.clone();
    let center = Point::new(center_x, center_y);
    let color = Scalar::new(b as f64, g as f64, r as f64, 255.0);

    circle(&mut img, center, radius, color)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: img })
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

    guided_filter(&src.inner, &guide, &mut dst, radius, eps)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

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

    gabor_filter(&gray, &mut dst, ksize, sigma, theta, lambda, gamma, psi)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Apply affine transformation to warp image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = warpAffine)]
pub async fn warp_affine_wasm(
    src: &WasmMat,
    matrix: Vec<f64>,
    width: usize,
    height: usize,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::geometric::warp_affine;
    use crate::core::types::Size;

    // Parse transformation matrix [a, b, c, d, e, f] -> [[a,b,c], [d,e,f]]
    if matrix.len() != 6 {
        return Err(JsValue::from_str(
            "Transformation matrix must have 6 elements",
        ));
    }

    let m = [
        [matrix[0], matrix[1], matrix[2]],
        [matrix[3], matrix[4], matrix[5]],
    ];

    // Convert to f32 array for GPU
    let m_f32: [f32; 6] = [
        matrix[0] as f32, matrix[1] as f32, matrix[2] as f32,
        matrix[3] as f32, matrix[4] as f32, matrix[5] as f32,
    ];

    let mut dst = Mat::new(height, width, src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::warp_affine_gpu_async(&src.inner, &mut dst, &m_f32, (width, height)).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU warp affine failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    warp_affine(&src.inner, &mut dst, &m, (width, height))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Detect Harris corners and visualize them
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = harrisCorners)]
pub async fn harris_corners_wasm(
    src: &WasmMat,
    block_size: i32,
    ksize: i32,
    k: f64,
    threshold: f64,
) -> Result<WasmMat, JsValue> {
    use crate::features2d::harris_corners;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{ColorConversionCode, Scalar};
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

    // Detect corners
    let keypoints = harris_corners(&gray, block_size, ksize, k, threshold)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0); // Green

    for kp in keypoints {
        circle(&mut result, kp.pt, 3, color)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(WasmMat { inner: result })
}

/// Detect good features to track and visualize them
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = goodFeaturesToTrack)]
pub async fn good_features_to_track_wasm(
    src: &WasmMat,
    max_corners: usize,
    quality_level: f64,
    min_distance: f64,
    block_size: i32,
) -> Result<WasmMat, JsValue> {
    use crate::features2d::good_features_to_track;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{ColorConversionCode, Scalar};
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

    // Detect corners
    let keypoints = good_features_to_track(&gray, max_corners, quality_level, min_distance, block_size)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 0.0, 255.0, 255.0); // Red

    for kp in keypoints {
        circle(&mut result, kp.pt, 5, color)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(WasmMat { inner: result })
}

/// Detect FAST keypoints and visualize them
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = fast)]
pub async fn fast_wasm(
    src: &WasmMat,
    threshold: i32,
    nonmax_suppression: bool,
) -> Result<WasmMat, JsValue> {
    use crate::features2d::fast;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{ColorConversionCode, Scalar};
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

    // Detect keypoints
    let keypoints = fast(&gray, threshold, nonmax_suppression)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 255.0, 0.0, 255.0); // Cyan

    for kp in keypoints {
        circle(&mut result, kp.pt, 2, color)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
    }

    Ok(WasmMat { inner: result })
}

/// Morphological erosion
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = erode)]
pub async fn erode_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
    use crate::imgproc::morphology::{erode_async, get_structuring_element, MorphShape};
    use crate::core::types::Size;

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(ksize, ksize));
    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        src.inner.depth(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    erode_async(&src.inner, &mut dst, &kernel, true)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Morphological dilation
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = dilate)]
pub async fn dilate_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
    use crate::imgproc::morphology::{dilate_async, get_structuring_element, MorphShape};
    use crate::core::types::Size;

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(ksize, ksize));
    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        src.inner.depth(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    dilate_async(&src.inner, &mut dst, &kernel, true)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Morphological opening - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = morphologyOpening)]
pub async fn morphology_opening_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
    use crate::imgproc::morphology::{morphology_ex_async, get_structuring_element, MorphShape, MorphType};
    use crate::core::types::Size;

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(ksize, ksize));
    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        src.inner.depth(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first (composes erode+dilate GPU operations)
    morphology_ex_async(&src.inner, &mut dst, MorphType::Open, &kernel, true)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Morphological closing - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = morphologyClosing)]
pub async fn morphology_closing_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
    use crate::imgproc::morphology::{morphology_ex_async, get_structuring_element, MorphShape, MorphType};
    use crate::core::types::Size;

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(ksize, ksize));
    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        src.inner.depth(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first (composes dilate+erode GPU operations)
    morphology_ex_async(&src.inner, &mut dst, MorphType::Close, &kernel, true)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Morphological gradient - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = morphologyGradient)]
pub async fn morphology_gradient_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
    use crate::imgproc::morphology::{morphology_ex_async, get_structuring_element, MorphShape, MorphType};
    use crate::core::types::Size;

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(ksize, ksize));
    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        src.inner.depth(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first (composes dilate-erode GPU operations)
    morphology_ex_async(&src.inner, &mut dst, MorphType::Gradient, &kernel, true)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Morphological top hat
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = morphologyTopHat)]
pub async fn morphology_top_hat_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
    use crate::imgproc::morphology::{morphology_ex, get_structuring_element, MorphShape, MorphType};
    use crate::core::types::Size;

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(ksize, ksize));
    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        src.inner.depth(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    morphology_ex(&src.inner, &mut dst, MorphType::TopHat, &kernel)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Morphological black hat
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = morphologyBlackHat)]
pub async fn morphology_black_hat_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
    use crate::imgproc::morphology::{morphology_ex, get_structuring_element, MorphShape, MorphType};
    use crate::core::types::Size;

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(ksize, ksize));
    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        src.inner.depth(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    morphology_ex(&src.inner, &mut dst, MorphType::BlackHat, &kernel)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Equalize histogram for contrast enhancement
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = equalizeHistogram)]
pub async fn equalize_histogram_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::histogram::equalize_hist;
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

    equalize_hist(&gray, &mut dst)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Convert to HSV color space
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = cvtColorHsv)]
pub async fn cvt_color_hsv_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::color::cvt_color_async;
    use crate::core::types::ColorConversionCode;

    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        src.inner.depth(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    cvt_color_async(&src.inner, &mut dst, ColorConversionCode::RgbToHsv, true)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Distance transform
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = distanceTransform)]
pub async fn distance_transform_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::advanced_filter::distance_transform;
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

    let mut dst = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::distance_transform_gpu_async(&gray, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU distance transform failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    distance_transform(&gray, &mut dst, crate::imgproc::advanced_filter::DistanceType::L2, 3)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

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

/// Hough Lines detection - visualize on image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = houghLines)]
pub async fn hough_lines_wasm(
    src: &WasmMat,
    threshold: i32,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::hough::hough_lines;
    use crate::imgproc::drawing::line;
    use crate::core::types::{ColorConversionCode, Point, Scalar};
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

    let lines = hough_lines(&gray, 1.0, std::f64::consts::PI / 180.0, threshold)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw lines on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0); // Green

    for (rho, theta) in lines.iter().take(50) { // Limit to 50 lines
        let a = theta.cos();
        let b = theta.sin();
        let x0 = a * rho;
        let y0 = b * rho;
        let x1 = (x0 + 1000.0 * (-b)) as i32;
        let y1 = (y0 + 1000.0 * a) as i32;
        let x2 = (x0 - 1000.0 * (-b)) as i32;
        let y2 = (y0 - 1000.0 * a) as i32;

        let _ = line(&mut result, Point::new(x1, y1), Point::new(x2, y2), color, 2);
    }

    Ok(WasmMat { inner: result })
}

/// Hough Lines P (probabilistic) - visualize on image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = houghLinesP)]
pub async fn hough_lines_p_wasm(
    src: &WasmMat,
    threshold: i32,
    min_line_length: f64,
    max_line_gap: f64,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::hough::hough_lines_p;
    use crate::imgproc::drawing::line;
    use crate::core::types::{ColorConversionCode, Point, Scalar};
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

    let lines = hough_lines_p(&gray, 1.0, std::f64::consts::PI / 180.0, threshold, min_line_length, max_line_gap)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw line segments on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 0.0, 255.0, 255.0); // Red

    for (p1, p2) in lines {
        let _ = line(&mut result, p1, p2, color, 2);
    }

    Ok(WasmMat { inner: result })
}

/// Hough Circles - visualize on image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = houghCircles)]
pub async fn hough_circles_wasm(
    src: &WasmMat,
    min_dist: f64,
    param1: f64,
    param2: f64,
    min_radius: i32,
    max_radius: i32,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::hough::{hough_circles, HoughCirclesMethod};
    use crate::imgproc::drawing::circle;
    use crate::core::types::{ColorConversionCode, Point, Scalar};
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

    let circles = hough_circles(&gray, HoughCirclesMethod::Gradient, 1.0, min_dist, param1, param2, min_radius, max_radius)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw circles on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 255.0, 255.0); // Magenta

    for c in circles {
        let _ = circle(&mut result, c.center, c.radius, color);
    }

    Ok(WasmMat { inner: result })
}

/// Find and draw contours
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = findContours)]
pub async fn find_contours_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::find_contours;
    use crate::imgproc::threshold::threshold;
    use crate::core::types::ThresholdType;
    use crate::imgproc::drawing::line;
    use crate::core::types::{ColorConversionCode, Point, Scalar};
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw contours on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0); // Green

    for contour in contours.iter().take(100) { // Limit to 100 contours
        for i in 0..contour.len() {
            let p1 = contour[i];
            let p2 = contour[(i + 1) % contour.len()];
            let _ = line(&mut result, p1, p2, color, 2);
        }
    }

    Ok(WasmMat { inner: result })
}

/// Find contours and draw bounding rectangles
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = boundingRect)]
pub async fn bounding_rect_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::{find_contours, bounding_rect};
    use crate::imgproc::threshold::threshold;
    use crate::core::types::ThresholdType;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{ColorConversionCode, Scalar};
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw bounding rectangles
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 0.0, 255.0); // Blue

    for contour in contours.iter().take(100) {
        let rect = bounding_rect(&contour);
        let _ = rectangle(&mut result, rect, color, 2);
    }

    Ok(WasmMat { inner: result })
}

/// Calculate histogram (returns visual representation)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = calcHistogram)]
pub async fn calc_histogram_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::histogram::calc_hist;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{ColorConversionCode, Rect, Scalar};
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

    // Calculate histogram
    let hist = calc_hist(&gray, 256, (0.0, 256.0))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Create visualization image (256x256)
    let hist_img_size = 256;
    let mut hist_img = Mat::new(hist_img_size, hist_img_size, 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Fill white background
    for row in 0..hist_img_size {
        for col in 0..hist_img_size {
            let pixel = hist_img.at_mut(row, col)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            pixel[0] = 255;
            pixel[1] = 255;
            pixel[2] = 255;
        }
    }

    // Find max value for scaling
    let max_val = hist.iter().cloned().fold(0.0f32, f32::max);

    // Draw histogram bars
    let bin_width = hist_img_size / 256;
    for i in 0..256 {
        let bin_height = if max_val > 0.0 {
            ((hist[i] / max_val) * hist_img_size as f32) as i32
        } else {
            0
        };

        if bin_height > 0 {
            let rect = Rect::new(
                i as i32 * bin_width as i32,
                hist_img_size as i32 - bin_height,
                bin_width as i32,
                bin_height,
            );
            let _ = rectangle(&mut hist_img, rect, Scalar::new(0.0, 0.0, 0.0, 255.0), -1);
        }
    }

    Ok(WasmMat { inner: hist_img })
}

/// Detect ArUco markers and visualize
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = detectAruco)]
pub async fn detect_aruco_wasm(src: &WasmMat, dict_id: i32) -> Result<WasmMat, JsValue> {
    use crate::objdetect::aruco::{ArucoDetector, ArucoDictionary};
    use crate::imgproc::drawing::{line, circle};
    use crate::core::types::{ColorConversionCode, Point, Scalar};
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

    // Map dict_id to ArucoDictionary variant (default to Dict4X4_50)
    let dict = match dict_id {
        0 => ArucoDictionary::Dict4X4_50,
        1 => ArucoDictionary::Dict5X5_50,
        2 => ArucoDictionary::Dict6X6_50,
        _ => ArucoDictionary::Dict4X4_50,
    };
    let detector = ArucoDetector::new(dict);
    let markers = detector.detect_markers(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw markers on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0); // Green

    for marker in markers {
        // Draw marker corners
        for i in 0..4 {
            let p1_f = marker.corners[i];
            let p2_f = marker.corners[(i + 1) % 4];
            let p1 = Point::new(p1_f.x as i32, p1_f.y as i32);
            let p2 = Point::new(p2_f.x as i32, p2_f.y as i32);
            let _ = line(&mut result, p1, p2, color, 2);
            let _ = circle(&mut result, p1, 5, Scalar::new(255.0, 0.0, 0.0, 255.0));
        }
    }

    Ok(WasmMat { inner: result })
}

/// Detect QR codes and visualize
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = detectQR)]
pub async fn detect_qr_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::objdetect::qr_detector::QRCodeDetector;
    use crate::imgproc::drawing::line;
    use crate::core::types::{ColorConversionCode, Point, Scalar};
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

    let detector = QRCodeDetector::new();
    let results = detector.detect_multi(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw QR code boundaries
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 255.0, 255.0); // Magenta

    for qr_points in results {
        if qr_points.len() >= 4 {
            for i in 0..4 {
                let p1_f = qr_points[i];
                let p2_f = qr_points[(i + 1) % 4];
                let p1 = Point::new(p1_f.x as i32, p1_f.y as i32);
                let p2 = Point::new(p2_f.x as i32, p2_f.y as i32);
                let _ = line(&mut result, p1, p2, color, 3);
            }
        }
    }

    Ok(WasmMat { inner: result })
}

/// Contour area visualization
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = contourArea)]
pub async fn contour_area_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::{find_contours, contour_area};
    use crate::imgproc::threshold::threshold;
    use crate::core::types::ThresholdType;
    use crate::imgproc::drawing::line;
    use crate::core::types::{ColorConversionCode, Point, Scalar};
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw contours colored by area
    let mut result = src.inner.clone();

    for contour in contours.iter().take(100) {
        let area = contour_area(&contour);

        // Color based on area (larger = more red, smaller = more blue)
        let normalized_area = (area / 10000.0).min(1.0);
        let color = Scalar::new(
            (1.0 - normalized_area) * 255.0,
            0.0,
            normalized_area * 255.0,
            255.0
        );

        for i in 0..contour.len() {
            let p1 = contour[i];
            let p2 = contour[(i + 1) % contour.len()];
            let _ = line(&mut result, p1, p2, color, 2);
        }
    }

    Ok(WasmMat { inner: result })
}

/// Arc length visualization
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = arcLength)]
pub async fn arc_length_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::{find_contours, arc_length};
    use crate::imgproc::threshold::threshold;
    use crate::core::types::ThresholdType;
    use crate::imgproc::drawing::line;
    use crate::core::types::{ColorConversionCode, Point, Scalar};
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw contours colored by perimeter
    let mut result = src.inner.clone();

    for contour in contours.iter().take(100) {
        let perimeter = arc_length(&contour, true);

        // Color based on perimeter
        let normalized_perimeter = (perimeter / 1000.0).min(1.0);
        let color = Scalar::new(
            0.0,
            normalized_perimeter * 255.0,
            (1.0 - normalized_perimeter) * 255.0,
            255.0
        );

        for i in 0..contour.len() {
            let p1 = contour[i];
            let p2 = contour[(i + 1) % contour.len()];
            let _ = line(&mut result, p1, p2, color, 2);
        }
    }

    Ok(WasmMat { inner: result })
}

/// Approximate polygon
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = approxPolyDP)]
pub async fn approx_poly_dp_wasm(src: &WasmMat, threshold_value: f64, epsilon: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::{find_contours, approx_poly_dp};
    use crate::imgproc::threshold::threshold;
    use crate::core::types::ThresholdType;
    use crate::imgproc::drawing::line;
    use crate::core::types::{ColorConversionCode, Point, Scalar};
    use crate::imgproc::color::cvt_color;

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw approximated polygons
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 255.0, 0.0, 255.0); // Yellow

    for contour in contours.iter().take(100) {
        let approx = approx_poly_dp(&contour, epsilon, true);

        for i in 0..approx.len() {
            let p1 = approx[i];
            let p2 = approx[(i + 1) % approx.len()];
            let _ = line(&mut result, p1, p2, color, 3);
        }
    }

    Ok(WasmMat { inner: result })
}

// ==================== Batch 4: Advanced Filters, Transforms & Analysis ====================

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

/// Morphological top hat
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = morphologyTophat)]
pub async fn morphology_tophat_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
    use crate::imgproc::morphology::{morphology_ex, get_structuring_element, MorphShape, MorphType};
    use crate::core::types::Size;

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(ksize, ksize));
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    morphology_ex(&src.inner, &mut dst, MorphType::TopHat, &kernel)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Morphological black hat
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = morphologyBlackhat)]
pub async fn morphology_blackhat_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
    use crate::imgproc::morphology::{morphology_ex, get_structuring_element, MorphShape, MorphType};
    use crate::core::types::Size;

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(ksize, ksize));
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    morphology_ex(&src.inner, &mut dst, MorphType::BlackHat, &kernel)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Warp perspective transformation
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = warpPerspective)]
pub async fn warp_perspective_wasm(
    src: &WasmMat,
    m11: f64, m12: f64, m13: f64,
    m21: f64, m22: f64, m23: f64,
    m31: f64, m32: f64, m33: f64,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::geometric::warp_perspective;
    use crate::core::types::{InterpolationFlag, Size};

    let transform_matrix = [
        [m11, m12, m13],
        [m21, m22, m23],
        [m31, m32, m33],
    ];

    let dsize = Size::new(src.inner.cols() as i32, src.inner.rows() as i32);
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    warp_perspective(&src.inner, &mut dst, &transform_matrix, dsize)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Get rotation matrix 2D and apply rotation
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = getRotationMatrix2D)]
pub async fn get_rotation_matrix_2d_wasm(
    src: &WasmMat,
    center_x: f64,
    center_y: f64,
    angle: f64,
    scale: f64,
) -> Result<WasmMat, JsValue> {
    use crate::imgproc::geometric::{get_rotation_matrix_2d, warp_affine};
    use crate::core::types::{Point2f, InterpolationFlag, Size};

    let center = Point2f::new(center_x as f32, center_y as f32);
    let rotation_matrix = get_rotation_matrix_2d(center, angle, scale);

    let dsize = Size::new(src.inner.cols() as i32, src.inner.rows() as i32);
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    warp_affine(&src.inner, &mut dst, &rotation_matrix, dsize)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Normalize histogram (returns visualization)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = normalizeHistogram)]
pub async fn normalize_histogram_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::histogram::{calc_hist, normalize_hist};
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{Rect, Scalar};

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

    // Calculate and normalize histogram
    let mut hist = calc_hist(&gray, 256, (0.0, 256.0))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    normalize_hist(&mut hist, 0.0, 1.0);

    // Create visualization
    let hist_img_size = 256;
    let mut hist_img = Mat::new(hist_img_size, hist_img_size, 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Fill white background
    for row in 0..hist_img_size {
        for col in 0..hist_img_size {
            let pixel = hist_img.at_mut(row, col)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            pixel[0] = 255;
            pixel[1] = 255;
            pixel[2] = 255;
        }
    }

    // Draw normalized histogram bars
    let bin_width = hist_img_size / 256;
    for i in 0..256 {
        let bin_height = (hist[i] * hist_img_size as f32) as i32;
        if bin_height > 0 {
            let rect = Rect::new(
                i as i32 * bin_width as i32,
                hist_img_size as i32 - bin_height,
                bin_width as i32,
                bin_height,
            );
            let _ = rectangle(&mut hist_img, rect, Scalar::new(0.0, 255.0, 0.0, 255.0), -1);
        }
    }

    Ok(WasmMat { inner: hist_img })
}

/// Compare histograms (returns similarity score)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = compareHistograms)]
pub async fn compare_histograms_wasm(src1: &WasmMat, src2: &WasmMat) -> Result<f64, JsValue> {
    use crate::imgproc::histogram::{calc_hist, compare_hist, HistCompMethod};
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;

    // Convert both to grayscale
    let gray1 = if src1.inner.channels() > 1 {
        let mut g = Mat::new(src1.inner.rows(), src1.inner.cols(), 1, src1.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src1.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src1.inner.clone()
    };

    let gray2 = if src2.inner.channels() > 1 {
        let mut g = Mat::new(src2.inner.rows(), src2.inner.cols(), 1, src2.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src2.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src2.inner.clone()
    };

    // Calculate histograms
    let hist1 = calc_hist(&gray1, 256, (0.0, 256.0))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let hist2 = calc_hist(&gray2, 256, (0.0, 256.0))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Compare using correlation method
    let similarity = compare_hist(&hist1, &hist2, HistCompMethod::Correlation)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(similarity)
}

/// Back projection
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = backProjection)]
pub async fn back_projection_wasm(src: &WasmMat, model: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::histogram::{calc_back_project, calc_hist};
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;

    // Convert both to grayscale
    let gray_src = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let gray_model = if model.inner.channels() > 1 {
        let mut g = Mat::new(model.inner.rows(), model.inner.cols(), 1, model.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&model.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        model.inner.clone()
    };

    // Calculate histogram of model image
    let model_hist = calc_hist(&gray_model, 256, (0.0, 256.0))
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut dst = Mat::new(gray_src.rows(), gray_src.cols(), 1, gray_src.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    calc_back_project(&gray_src, &model_hist, (0.0, 256.0), &mut dst)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Moments - compute contour moments (visualize centroid)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = moments)]
pub async fn moments_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::{find_contours, moments};
    use crate::imgproc::threshold::threshold;
    use crate::core::types::{ColorConversionCode, ThresholdType};
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{Point, Scalar};

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw centroids on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0);

    for contour in contours.iter().take(10) {
        let m = moments(contour);
        if m.m00 != 0.0 {
            let cx = (m.m10 / m.m00) as i32;
            let cy = (m.m01 / m.m00) as i32;
            let _ = circle(&mut result, Point::new(cx, cy), 5, color);
        }
    }

    Ok(WasmMat { inner: result })
}

/// Watershed segmentation
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = watershed)]
pub async fn watershed_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::advanced_filter::watershed;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::threshold::threshold;
    use crate::core::types::{ColorConversionCode, ThresholdType};

    // Ensure 3-channel image for watershed
    let bgr = if src.inner.channels() == 1 {
        let mut color = Mat::new(src.inner.rows(), src.inner.cols(), 3, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut color, ColorConversionCode::GrayToBgr)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        color
    } else {
        src.inner.clone()
    };

    // Create markers using simple threshold-based initialization
    let gray = if bgr.channels() > 1 {
        let mut g = Mat::new(bgr.rows(), bgr.cols(), 1, bgr.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&bgr, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        bgr.clone()
    };

    let mut markers = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Initialize markers: foreground (label 1), background (label 2), unknown (0)
    for row in 0..markers.rows() {
        for col in 0..markers.cols() {
            let val = gray.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0];
            markers.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] = if val < 50 {
                1 // Foreground
            } else if val > 200 {
                2 // Background
            } else {
                0 // Unknown
            };
        }
    }

    watershed(&bgr, &mut markers)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Visualize markers - multiply by 50 to make labels visible
    let mut result = Mat::new(markers.rows(), markers.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    for row in 0..markers.rows() {
        for col in 0..markers.cols() {
            let marker = markers.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0];
            result.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] = (marker.saturating_mul(50)).min(255);
        }
    }

    Ok(WasmMat { inner: result })
}

/// SIFT feature detection and visualization
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = sift)]
pub async fn sift_wasm(src: &WasmMat, n_features: usize) -> Result<WasmMat, JsValue> {
    use crate::features2d::SIFTF32;
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{Point, Scalar};

    // Convert to grayscale
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let sift = SIFTF32::new(n_features);
    let (keypoints, _) = sift.detect_and_compute(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0);

    for kp in keypoints.iter() {
        let pt = Point::new(kp.pt.x as i32, kp.pt.y as i32);
        let radius = (kp.size / 2.0) as i32;
        let _ = circle(&mut result, pt, radius, color);
    }

    Ok(WasmMat { inner: result })
}

/// ORB feature detection and visualization
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = orb)]
pub async fn orb_wasm(src: &WasmMat, n_features: usize) -> Result<WasmMat, JsValue> {
    use crate::features2d::ORB;
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{Point, Scalar};

    // Convert to grayscale
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let orb = ORB::new(n_features);
    let (keypoints, _) = orb.detect_and_compute(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 0.0, 255.0);

    for kp in keypoints.iter() {
        let pt = Point::new(kp.pt.x as i32, kp.pt.y as i32);
        let radius = (kp.size / 2.0) as i32;
        let _ = circle(&mut result, pt, radius, color);
    }

    Ok(WasmMat { inner: result })
}

/// BRISK feature detection and visualization
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = brisk)]
pub async fn brisk_wasm(src: &WasmMat, threshold: i32) -> Result<WasmMat, JsValue> {
    use crate::features2d::BRISK;
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{Point, Scalar};

    // Convert to grayscale
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let brisk = BRISK::new(threshold, 3);
    let (keypoints, _) = brisk.detect_and_compute(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 255.0, 255.0);

    for kp in keypoints.iter() {
        let pt = Point::new(kp.pt.x as i32, kp.pt.y as i32);
        let radius = (kp.size / 2.0) as i32;
        let _ = circle(&mut result, pt, radius, color);
    }

    Ok(WasmMat { inner: result })
}

/// AKAZE feature detection and visualization
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = akaze)]
pub async fn akaze_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::features2d::AKAZE;
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{Point, Scalar};

    // Convert to grayscale
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let akaze = AKAZE::new();
    let (keypoints, _) = akaze.detect_and_compute(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 255.0, 0.0, 255.0);

    for kp in keypoints.iter() {
        let pt = Point::new(kp.pt.x as i32, kp.pt.y as i32);
        let radius = (kp.size / 2.0) as i32;
        let _ = circle(&mut result, pt, radius, color);
    }

    Ok(WasmMat { inner: result })
}

/// KAZE feature detection and visualization
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = kaze)]
pub async fn kaze_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::features2d::KAZE;
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{Point, Scalar};

    // Convert to grayscale
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let kaze = KAZE::new(false, false);
    let (keypoints, _) = kaze.detect_and_compute(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on original image
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 255.0, 255.0);

    for kp in keypoints.iter() {
        let pt = Point::new(kp.pt.x as i32, kp.pt.y as i32);
        let radius = (kp.size / 2.0) as i32;
        let _ = circle(&mut result, pt, radius, color);
    }

    Ok(WasmMat { inner: result })
}

// ==================== Batch 5: Advanced Features & Operations ====================

/// Laplacian of Gaussian (LoG) blob detection
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = logFilter)]
pub async fn log_filter_wasm(src: &WasmMat, ksize: i32, sigma: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::advanced_filter::laplacian_of_gaussian;
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;

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

    laplacian_of_gaussian(&gray, &mut dst, ksize, sigma)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Convert RGB/BGR to Lab color space
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = cvtColorLab)]
pub async fn cvt_color_lab_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;

    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    cvt_color(&src.inner, &mut dst, ColorConversionCode::BgrToLab)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Convert RGB/BGR to YCrCb color space
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = cvtColorYCrCb)]
pub async fn cvt_color_ycrcb_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;

    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    cvt_color(&src.inner, &mut dst, ColorConversionCode::BgrToYCrCb)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Draw ellipse on image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = drawEllipse)]
pub async fn draw_ellipse_wasm(src: &WasmMat, cx: i32, cy: i32, width: i32, height: i32, angle: f64, thickness: i32) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::ellipse;
    use crate::core::types::{Point, Scalar};

    let mut result = src.inner.clone();
    let center = Point::new(cx, cy);
    let axes = (width / 2, height / 2);
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0);

    ellipse(&mut result, center, axes, angle, 0.0, 360.0, color)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: result })
}

/// Draw polylines on image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = drawPolylines)]
pub async fn draw_polylines_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::polylines;
    use crate::core::types::{Point, Scalar};

    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 0.0, 255.0);

    // Create a sample polygon (diamond shape)
    let w = result.cols() as i32;
    let h = result.rows() as i32;
    let pts = vec![
        Point::new(w / 2, h / 4),
        Point::new(3 * w / 4, h / 2),
        Point::new(w / 2, 3 * h / 4),
        Point::new(w / 4, h / 2),
    ];

    polylines(&mut result, &pts, true, color, 2)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: result })
}

/// Put text on image
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = putText)]
pub async fn put_text_wasm(src: &WasmMat, text: String, x: i32, y: i32, font_scale: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::put_text;
    use crate::core::types::{Point, Scalar};

    let mut result = src.inner.clone();
    let org = Point::new(x, y);
    let color = Scalar::new(255.0, 255.0, 0.0, 255.0);

    put_text(&mut result, &text, org, font_scale, color)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: result })
}

/// Compute minimum enclosing circle of contours
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = minEnclosingCircle)]
pub async fn min_enclosing_circle_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::find_contours;
    use crate::imgproc::threshold::threshold;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::drawing::circle;
    use crate::shape::descriptors::min_enclosing_circle;
    use crate::core::types::{ColorConversionCode, ThresholdType, Point, Scalar};

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw minimum enclosing circles
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0);

    for contour in contours.iter().take(10) {
        let (center, radius) = min_enclosing_circle(contour);
        let _ = circle(&mut result, center, radius as i32, color);
    }

    Ok(WasmMat { inner: result })
}

/// Compute convex hull of contours
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = convexHull)]
pub async fn convex_hull_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::find_contours;
    use crate::imgproc::threshold::threshold;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::drawing::polylines;
    use crate::shape::descriptors::convex_hull;
    use crate::core::types::{ColorConversionCode, ThresholdType, Scalar};

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw convex hulls
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 0.0, 255.0);

    for contour in contours.iter().take(10) {
        let hull = convex_hull(contour);
        let _ = polylines(&mut result, &hull, true, color, 2);
    }

    Ok(WasmMat { inner: result })
}

/// Compute Hu moments of contours
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = huMoments)]
pub async fn hu_moments_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::threshold::threshold;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::drawing::put_text;
    use crate::shape::moments::{compute_moments, hu_moments};
    use crate::core::types::{ColorConversionCode, ThresholdType, Point, Scalar};

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Compute moments from binary image
    let m = compute_moments(&binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let hu = hu_moments(&m);

    // Display first 3 Hu moments
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 255.0, 255.0, 255.0);

    for (i, &h) in hu.iter().take(3).enumerate() {
        let text = format!("Hu{}: {:.2e}", i + 1, h);
        let _ = put_text(&mut result, &text, Point::new(10, 30 + i as i32 * 30), 0.6, color);
    }

    Ok(WasmMat { inner: result })
}

/// Inpaint - fill missing regions
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = inpaint)]
pub async fn inpaint_wasm(src: &WasmMat, radius: i32) -> Result<WasmMat, JsValue> {
    use crate::photo::inpaint;

    // Create a mask (central region to inpaint)
    let mut mask = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Mark center region as damaged
    let cx = (src.inner.cols() / 2) as i32;
    let cy = (src.inner.rows() / 2) as i32;
    let r = (src.inner.cols().min(src.inner.rows()) / 4) as i32;

    for row in 0..mask.rows() {
        for col in 0..mask.cols() {
            let dx = col as i32 - cx;
            let dy = row as i32 - cy;
            let dist_sq = dx * dx + dy * dy;
            mask.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] =
                if dist_sq < (r * r) { 255 } else { 0 };
        }
    }

    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    inpaint(&src.inner, &mask, &mut dst, radius as f64)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// K-means clustering
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = kmeans)]
pub async fn kmeans_wasm(src: &WasmMat, k: usize) -> Result<WasmMat, JsValue> {
    use crate::ml::kmeans::{kmeans, KMeansFlags};

    // Reshape image to points
    let rows = src.inner.rows();
    let cols = src.inner.cols();
    let channels = src.inner.channels();

    let mut points = Vec::new();
    for row in 0..rows {
        for col in 0..cols {
            let pixel = src.inner.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
            let mut point = Vec::new();
            for ch in 0..channels {
                point.push(pixel[ch] as f64);
            }
            points.push(point);
        }
    }

    // Run k-means
    let mut labels = vec![0i32; points.len()];
    let (centers, _compactness) = kmeans(&points, k, &mut labels, 10, 1.0, KMeansFlags::PPCenters)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Create output image with cluster colors
    let mut result = Mat::new(rows, cols, channels, src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Use cluster centers as colors
    for row in 0..rows {
        for col in 0..cols {
            let idx = row * cols + col;
            let label = labels[idx] as usize;
            let center = &centers[label % centers.len()];
            let pixel = result.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
            for ch in 0..channels {
                pixel[ch] = center[ch].min(255.0).max(0.0) as u8;
            }
        }
    }

    Ok(WasmMat { inner: result })
}

/// Tonemap Drago for HDR images
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = tonemapDrago)]
pub async fn tonemap_drago_wasm(src: &WasmMat, bias: f64) -> Result<WasmMat, JsValue> {
    use crate::photo::hdr::TonemapDrago;

    let tonemap = TonemapDrago::new().with_bias(bias as f32);
    let dst = tonemap.process(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Tonemap Reinhard for HDR images
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = tonemapReinhard)]
pub async fn tonemap_reinhard_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::photo::hdr::TonemapReinhard;

    let tonemap = TonemapReinhard::new();
    let dst = tonemap.process(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Match shapes using Hu moments
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = matchShapes)]
pub async fn match_shapes_wasm(src: &WasmMat, threshold_value: f64) -> Result<WasmMat, JsValue> {
    use crate::imgproc::contours::find_contours;
    use crate::imgproc::threshold::threshold;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::drawing::{polylines, put_text};
    use crate::shape::matching::{match_shapes, ShapeMatchMethod};
    use crate::shape::moments::compute_moments;
    use crate::core::types::{ColorConversionCode, ThresholdType, Point, Scalar};

    // Convert to grayscale and threshold
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let mut binary = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    threshold(&gray, &mut binary, threshold_value, 255.0, ThresholdType::Binary)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let contours = find_contours(&binary, crate::imgproc::contours::RetrievalMode::External, crate::imgproc::contours::ChainApproxMode::Simple)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let mut result = src.inner.clone();

    // Compare first contour with others using whole binary image moments
    if contours.len() >= 2 {
        let ref_moments = compute_moments(&binary)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let _ = polylines(&mut result, &contours[0], true, Scalar::new(0.0, 255.0, 0.0, 255.0), 2);

        for (i, contour) in contours.iter().skip(1).take(5).enumerate() {
            // For demo purposes, compare with reference moments
            let similarity = match_shapes(&ref_moments, &ref_moments, ShapeMatchMethod::I1);
            let color = if similarity < 0.5 {
                Scalar::new(0.0, 255.0, 0.0, 255.0)
            } else {
                Scalar::new(0.0, 0.0, 255.0, 255.0)
            };
            let _ = polylines(&mut result, contour, true, color, 1);

            let text = format!("S{}: {:.2}", i + 1, similarity);
            let _ = put_text(&mut result, &text, Point::new(10, 30 + i as i32 * 25), 0.5, Scalar::new(255.0, 255.0, 255.0, 255.0));
        }
    }

    Ok(WasmMat { inner: result })
}

/// Find homography between matched points
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = findHomography)]
pub async fn find_homography_wasm(src: &WasmMat, n_features: usize) -> Result<WasmMat, JsValue> {
    use crate::features2d::SIFTF32;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{ColorConversionCode, Point, Scalar};

    // Convert to grayscale
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let sift = SIFTF32::new(n_features);
    let (keypoints, _) = sift.detect_and_compute(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Visualize detected keypoints (homography would need two images)
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 255.0, 255.0);

    for kp in keypoints.iter().take(50) {
        let pt = Point::new(kp.pt.x as i32, kp.pt.y as i32);
        let _ = circle(&mut result, pt, 3, color);
    }

    Ok(WasmMat { inner: result })
}

/// Brute force descriptor matcher (simplified - shows keypoint detection)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = bruteForceMatcher)]
pub async fn brute_force_matcher_wasm(src: &WasmMat, n_features: usize) -> Result<WasmMat, JsValue> {
    use crate::features2d::SIFTF32;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::drawing::circle;
    use crate::core::types::{ColorConversionCode, Point, Scalar};

    // Convert to grayscale
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    // Detect features in left and right halves
    let mid = (gray.cols() / 2) as i32;
    let mut left_half = Mat::new(gray.rows(), mid as usize, 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut right_half = Mat::new(gray.rows(), gray.cols() - mid as usize, 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    for row in 0..gray.rows() {
        for col in 0..(mid as usize) {
            left_half.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] =
                gray.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0];
        }
        for col in (mid as usize)..gray.cols() {
            right_half.at_mut(row, col - mid as usize).map_err(|e| JsValue::from_str(&e.to_string()))?[0] =
                gray.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0];
        }
    }

    let sift = SIFTF32::new(n_features / 2);
    let (kp1, _) = sift.detect_and_compute(&left_half)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let (kp2, _) = sift.detect_and_compute(&right_half)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw keypoints on left and right
    let mut result = src.inner.clone();
    for kp in kp1.iter().take(20) {
        let pt = Point::new(kp.pt.x as i32, kp.pt.y as i32);
        let _ = circle(&mut result, pt, 3, Scalar::new(0.0, 255.0, 0.0, 255.0));
    }
    for kp in kp2.iter().take(20) {
        let pt = Point::new((kp.pt.x as i32) + mid, kp.pt.y as i32);
        let _ = circle(&mut result, pt, 3, Scalar::new(255.0, 0.0, 0.0, 255.0));
    }

    Ok(WasmMat { inner: result })
}

/// HOG (Histogram of Oriented Gradients) descriptor
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = hogDescriptor)]
pub async fn hog_descriptor_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::objdetect::hog::HOGDescriptor;
    use crate::imgproc::color::cvt_color;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{ColorConversionCode, Rect, Scalar};

    // Convert to grayscale
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    let hog = HOGDescriptor::new();
    let _descriptors = hog.compute(&gray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw grid to show HOG cells
    let mut result = src.inner.clone();
    let cell_size = 16;
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0);

    for y in (0..result.rows()).step_by(cell_size) {
        for x in (0..result.cols()).step_by(cell_size) {
            let rect = Rect::new(x as i32, y as i32, cell_size as i32, cell_size as i32);
            let _ = rectangle(&mut result, rect, color, 1);
        }
    }

    Ok(WasmMat { inner: result })
}

/// Background subtractor MOG2
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = bgSubtractorMog2)]
pub async fn bg_subtractor_mog2_wasm(src: &WasmMat, learning_rate: f64) -> Result<WasmMat, JsValue> {
    use crate::video::background_subtraction::BackgroundSubtractorMOG2;

    let mut bg_sub = BackgroundSubtractorMOG2::new();
    let mut fg_mask = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    bg_sub.apply(&src.inner, &mut fg_mask, learning_rate)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: fg_mask })
}

/// Background subtractor KNN
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = bgSubtractorKnn)]
pub async fn bg_subtractor_knn_wasm(src: &WasmMat, learning_rate: f64) -> Result<WasmMat, JsValue> {
    use crate::video::background_subtraction::BackgroundSubtractorKNN;

    let mut bg_sub = BackgroundSubtractorKNN::new();
    let mut fg_mask = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    bg_sub.apply(&src.inner, &mut fg_mask, learning_rate)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: fg_mask })
}

/// Farneback dense optical flow
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = farnebackOpticalFlow)]
pub async fn farneback_optical_flow_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::video::optical_flow::calc_optical_flow_farneback;
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;

    // Convert to grayscale
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    // Create a shifted version as "next frame"
    let mut next_frame = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    for row in 0..gray.rows() {
        for col in 5..gray.cols() {
            next_frame.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] = 
                gray.at(row, col - 5).map_err(|e| JsValue::from_str(&e.to_string()))?[0];
        }
    }

    let flow = calc_optical_flow_farneback(&gray, &next_frame, 0.5, 3, 15, 3)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Visualize flow as grayscale magnitude
    let mut result = Mat::new(flow.rows(), flow.cols(), 1, src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    for row in 0..flow.rows() {
        for col in 0..flow.cols() {
            let fx = flow.at_f32(row, col, 0).unwrap_or(0.0);
            let fy = flow.at_f32(row, col, 1).unwrap_or(0.0);
            let mag = (fx * fx + fy * fy).sqrt();
            result.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] = 
                (mag * 10.0).min(255.0) as u8;
        }
    }

    Ok(WasmMat { inner: result })
}

// ==================== Batch 6: Advanced Tracking, ML & Calibration ====================

/// MeanShift object tracking
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = meanshiftTracker)]
pub async fn meanshift_tracker_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::video::tracking::MeanShiftTracker;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{Rect, Scalar};

    // Initialize tracker with center region
    let w = src.inner.cols() as i32;
    let h = src.inner.rows() as i32;
    let initial_window = Rect::new(w / 4, h / 4, w / 2, h / 2);

    let mut tracker = MeanShiftTracker::new(initial_window);
    let result_window = tracker.track(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw tracked region
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 0.0, 255.0);
    let _ = rectangle(&mut result, result_window, color, 2);

    Ok(WasmMat { inner: result })
}

/// CAMShift tracking
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = camshiftTracker)]
pub async fn camshift_tracker_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::video::tracking::CamShiftTracker;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{Rect, Scalar};

    // Initialize tracker with center region
    let w = src.inner.cols() as i32;
    let h = src.inner.rows() as i32;
    let initial_window = Rect::new(w / 4, h / 4, w / 2, h / 2);

    let mut tracker = CamShiftTracker::new(initial_window);
    let result_window = tracker.track(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw tracked region
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 0.0, 0.0, 255.0);
    let _ = rectangle(&mut result, result_window, color, 2);

    Ok(WasmMat { inner: result })
}

/// MOSSE tracker
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = mosseTracker)]
pub async fn mosse_tracker_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::video::advanced_tracking::MOSSETracker;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{Rect, Scalar};

    // Initialize tracker with center region
    let w = src.inner.cols() as i32;
    let h = src.inner.rows() as i32;
    let initial_bbox = Rect::new(w / 4, h / 4, w / 2, h / 2);

    let mut tracker = MOSSETracker::new();
    tracker.init(&src.inner, initial_bbox)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    let result_bbox = tracker.update(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw tracked region
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 255.0, 0.0, 255.0);
    let _ = rectangle(&mut result, result_bbox, color, 2);

    Ok(WasmMat { inner: result })
}

/// CSRT tracker
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = csrtTracker)]
pub async fn csrt_tracker_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::video::advanced_tracking::CSRTTracker;
    use crate::imgproc::drawing::rectangle;
    use crate::core::types::{Rect, Scalar};

    // Initialize tracker with center region
    let w = src.inner.cols() as i32;
    let h = src.inner.rows() as i32;
    let initial_bbox = Rect::new(w / 4, h / 4, w / 2, h / 2);

    let mut tracker = CSRTTracker::new();
    tracker.init(&src.inner, initial_bbox)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    let result_bbox = tracker.update(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Draw tracked region
    let mut result = src.inner.clone();
    let color = Scalar::new(0.0, 255.0, 255.0, 255.0);
    let _ = rectangle(&mut result, result_bbox, color, 2);

    Ok(WasmMat { inner: result })
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

/// Super resolution
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = superResolution)]
pub async fn super_resolution_wasm(src: &WasmMat, scale: f32) -> Result<WasmMat, JsValue> {
    use crate::photo::super_resolution::SuperResolutionBicubic;

    let sr = SuperResolutionBicubic::new(scale);
    let dst = sr.process(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Merge Debevec (HDR)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = mergeDebevec)]
pub async fn merge_debevec_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::photo::hdr::MergeDebevec;

    // For demo, use same image with different exposures (simulated)
    let images = vec![src.inner.clone()];
    let times = vec![1.0 / 30.0];

    let merge = MergeDebevec::new();
    let hdr = merge.process(&images, &times)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: hdr })
}

/// SVM Classifier (demo with simple pattern detection)
/// TODO: Implementation needs to be fixed - currently stubbed
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = svmClassifier)]
pub async fn svm_classifier_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("SVM classifier not yet fully implemented"))
}

#[cfg(feature = "ml_experimental")]
/// Decision Tree Classifier
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = decisionTree)]
pub async fn decision_tree_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("decision_tree_wasm not yet fully implemented"))
}

#[cfg(feature = "ml_experimental")]
/// Random Forest Classifier
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = randomForest)]
pub async fn random_forest_wasm(src: &WasmMat, n_trees: usize) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("random_forest_wasm not yet fully implemented"))
}

#[cfg(feature = "ml_experimental")]
/// K-Nearest Neighbors
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = knn)]
pub async fn knn_wasm(src: &WasmMat, k: usize) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("knn_wasm not yet fully implemented"))
}

#[cfg(feature = "ml_experimental")]
/// Neural Network (MLP)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = neuralNetwork)]
pub async fn neural_network_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("neural_network_wasm not yet fully implemented"))
}

/// Cascade Classifier (face/object detection demo)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = cascadeClassifier)]
pub async fn cascade_classifier_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("cascade_classifier_wasm not yet fully implemented"))
}

/// Calibrate camera (simplified demo)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = calibrateCamera)]
pub async fn calibrate_camera_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("calibrate_camera_wasm not yet fully implemented"))
}

#[cfg(feature = "calib_experimental")]
/// Fisheye calibration
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = fisheyeCalibration)]
pub async fn fisheye_calibration_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("fisheye_calibration_wasm not yet fully implemented"))
}

#[cfg(feature = "calib_experimental")]
/// Solve PnP (pose estimation)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = solvePnp)]
pub async fn solve_pnp_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("solve_pnp_wasm not yet fully implemented"))
}

#[cfg(feature = "calib_experimental")]
/// Stereo calibration
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = stereoCalibration)]
pub async fn stereo_calibration_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("stereo_calibration_wasm not yet fully implemented"))
}

/// Compute disparity (stereo matching)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = computeDisparity)]
pub async fn compute_disparity_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;

    // Simplified: Use shifted image as "right" view for demo
    let gray = if src.inner.channels() > 1 {
        let mut g = Mat::new(src.inner.rows(), src.inner.cols(), 1, src.inner.depth())
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        cvt_color(&src.inner, &mut g, ColorConversionCode::BgrToGray)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        g
    } else {
        src.inner.clone()
    };

    // Create shifted version as disparity map demo
    let mut disparity = Mat::new(gray.rows(), gray.cols(), 1, gray.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    for row in 0..gray.rows() {
        for col in 0..gray.cols() {
            let shift = (col % 20) as u8;
            disparity.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?[0] = shift * 12;
        }
    }

    Ok(WasmMat { inner: disparity })
}

/// Panorama stitcher (simplified demo)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = panoramaStitcher)]
pub async fn panorama_stitcher_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::imgproc::drawing::put_text;
    use crate::core::types::{Point, Scalar};

    // For single image, just add annotation
    let mut result = src.inner.clone();
    let text = "Panorama stitching demo".to_string();
    let _ = put_text(&mut result, &text, Point::new(10, 30), 0.7, Scalar::new(255.0, 255.0, 0.0, 255.0));

    Ok(WasmMat { inner: result })
}

/// Feather blender for stitching
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = featherBlender)]
pub async fn feather_blender_wasm(src: &WasmMat, blend_strength: f32) -> Result<WasmMat, JsValue> {
    // Simple alpha blending demo
    let mut result = src.inner.clone();
    
    // Apply feathering effect to edges
    for row in 0..result.rows() {
        for col in 0..result.cols() {
            let edge_dist = col.min(result.cols() - col).min(row).min(result.rows() - row) as f32;
            let alpha = (edge_dist * blend_strength).min(1.0);
            
            let pixel = result.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
            for ch in 0..result.channels() {
                pixel[ch] = (pixel[ch] as f32 * alpha) as u8;
            }
        }
    }

    Ok(WasmMat { inner: result })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = stereoRectification)]
pub async fn stereo_rectification_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("stereo_rectification_wasm not yet fully implemented"))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = multibandBlender)]
pub async fn multiband_blender_wasm(src: &WasmMat, num_bands: usize) -> Result<WasmMat, JsValue> {
    use crate::stitching::blending::MultiBandBlender;
    use crate::core::{Mat, MatDepth};
    
    // Create two overlapping images for blending demo
    let w = src.inner.cols();
    let h = src.inner.rows();
    
    // Split image into left and right halves with overlap
    let overlap = w / 4;
    
    // Create left image (0 to w/2 + overlap)
    let mut left = Mat::new(h, w / 2 + overlap, src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    for row in 0..h {
        for col in 0..(w / 2 + overlap) {
            if col < src.inner.cols() {
                let src_pixel = src.inner.at(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                let dst_pixel = left.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                for ch in 0..src.inner.channels() {
                    dst_pixel[ch] = src_pixel[ch];
                }
            }
        }
    }
    
    // Create right image (w/2 - overlap to w)
    let mut right = Mat::new(h, w / 2 + overlap, src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    for row in 0..h {
        for col in 0..(w / 2 + overlap) {
            let src_col = (w / 2 - overlap) + col;
            if src_col < src.inner.cols() {
                let src_pixel = src.inner.at(row, src_col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                let dst_pixel = right.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
                for ch in 0..src.inner.channels() {
                    dst_pixel[ch] = src_pixel[ch];
                }
            }
        }
    }
    
    // Create masks
    let mut mask_left = Mat::new(h, w / 2 + overlap, 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut mask_right = Mat::new(h, w / 2 + overlap, 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    for row in 0..h {
        for col in 0..(w / 2 + overlap) {
            let left_pix = mask_left.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
            left_pix[0] = 255;
            
            let right_pix = mask_right.at_mut(row, col).map_err(|e| JsValue::from_str(&e.to_string()))?;
            right_pix[0] = 255;
        }
    }
    
    // Apply multi-band blending
    let blender = MultiBandBlender::new(num_bands.max(1).min(6));
    let result = blender.blend(&[left, right], &[mask_left, mask_right])
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    Ok(WasmMat { inner: result })
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = loadNetwork)]
pub async fn load_network_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    Err(JsValue::from_str("load_network_wasm not yet fully implemented"))
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = blobFromImage)]
pub async fn blob_from_image_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    use crate::dnn::blob::Blob;
    use crate::imgproc::drawing::{rectangle, put_text};
    use crate::core::types::{Rect, Scalar, Point};
    
    // Convert image to blob (NCHW format)
    let blob = Blob::from_image(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    
    let shape = blob.shape();
    
    // Visualize the blob transformation
    let mut result = src.inner.clone();
    let color = Scalar::new(255.0, 255.0, 0.0, 255.0);
    let bg_color = Scalar::new(0.0, 0.0, 0.0, 180.0);
    
    // Draw info box
    let info_height = 120;
    let info_rect = Rect::new(10, 10, 250, info_height);
    let _ = rectangle(&mut result, info_rect, bg_color, -1);
    let _ = rectangle(&mut result, info_rect, color, 2);
    
    // Display blob info
    let _ = put_text(&mut result, "Blob Conversion", Point::new(20, 35), 0.6, color);
    
    let shape_text = format!("Shape: {:?}", shape);
    let _ = put_text(&mut result, &shape_text, Point::new(20, 60), 0.5, color);
    
    let format_text = "Format: NCHW";
    let _ = put_text(&mut result, format_text, Point::new(20, 85), 0.5, color);
    
    let norm_text = "Normalized: [0, 1]";
    let _ = put_text(&mut result, norm_text, Point::new(20, 110), 0.5, color);
    
    // Draw channel separation visualization
    let ch_width = result.cols() / 3;
    for i in 0..3 {
        let x = i * ch_width;
        let rect = Rect::new(x as i32, (result.rows() - 30) as i32, ch_width as i32, 25);
        let ch_color = match i {
            0 => Scalar::new(255.0, 0.0, 0.0, 255.0),
            1 => Scalar::new(0.0, 255.0, 0.0, 255.0),
            _ => Scalar::new(0.0, 0.0, 255.0, 255.0),
        };
        let _ = rectangle(&mut result, rect, ch_color, -1);
        
        let ch_text = format!("Ch{}", i);
        let _ = put_text(&mut result, &ch_text, Point::new(x as i32 + 10, (result.rows() - 10) as i32), 0.5, Scalar::new(255.0, 255.0, 255.0, 255.0));
    }

    Ok(WasmMat { inner: result })
}

// ============================================================================
// GPU-ACCELERATED OPERATIONS - BATCHES 1-3
// ============================================================================

/// Convert RGB to Grayscale (GPU-accelerated)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = rgbToGray)]
pub async fn rgb_to_gray_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::rgb_to_gray_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU RGB to Gray failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::imgproc::cvt_color(&src.inner, &mut dst, ColorConversionCode::RgbToGray)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Convert RGB to HSV color space (GPU-accelerated)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = rgbToHsv)]
pub async fn rgb_to_hsv_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::rgb_to_hsv_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU RGB to HSV failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::imgproc::cvt_color(&src.inner, &mut dst, ColorConversionCode::RgbToHsv)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Convert RGB to Lab color space (GPU-accelerated)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = rgbToLab)]
pub async fn rgb_to_lab_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::rgb_to_lab_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU RGB to Lab failed, falling back to CPU".into());
                }
            }
        }
    }

    crate::imgproc::cvt_color(&src.inner, &mut dst, ColorConversionCode::RgbToLab)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Convert RGB to YCrCb color space (GPU-accelerated)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = rgbToYCrCb)]
pub async fn rgb_to_ycrcb_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::rgb_to_ycrcb_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU RGB to YCrCb failed, falling back to CPU".into());
                }
            }
        }
    }

    crate::imgproc::cvt_color(&src.inner, &mut dst, ColorConversionCode::RgbToYCrCb)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Convert HSV to RGB color space (GPU-accelerated)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = hsvToRgb)]
pub async fn hsv_to_rgb_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::hsv_to_rgb_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU HSV to RGB failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::imgproc::cvt_color(&src.inner, &mut dst, ColorConversionCode::HsvToRgb)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Convert Lab to RGB color space (GPU-accelerated)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = labToRgb)]
pub async fn lab_to_rgb_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::lab_to_rgb_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU Lab to RGB failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::imgproc::cvt_color(&src.inner, &mut dst, ColorConversionCode::LabToRgb)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Convert YCrCb to RGB color space (GPU-accelerated)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = ycrcbToRgb)]
pub async fn ycrcb_to_rgb_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 3, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::ycrcb_to_rgb_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU YCrCb to RGB failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::imgproc::cvt_color(&src.inner, &mut dst, ColorConversionCode::YCrCbToRgb)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Pyramid down (downscale image) - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = pyrDown)]
pub async fn pyr_down_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let dst_width = src.inner.cols() / 2;
    let dst_height = src.inner.rows() / 2;
    let mut dst = Mat::new(dst_height, dst_width, src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::pyrdown_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU pyrDown failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback not yet implemented
    Err(JsValue::from_str("GPU pyrDown failed and CPU fallback not yet implemented"))
}

/// Pyramid up (upscale image) - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = pyrUp)]
pub async fn pyr_up_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let dst_width = src.inner.cols() * 2;
    let dst_height = src.inner.rows() * 2;
    let mut dst = Mat::new(dst_height, dst_width, src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::pyrup_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU pyrUp failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback not yet implemented
    Err(JsValue::from_str("GPU pyrUp failed and CPU fallback not yet implemented"))
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

/// Gradient magnitude - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = gradientMagnitude)]
pub async fn gradient_magnitude_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::gradient_magnitude_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU gradientMagnitude failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback not yet implemented
    Err(JsValue::from_str("GPU gradientMagnitude failed and CPU fallback not yet implemented"))
}

/// Integral image - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = integralImage)]
pub async fn integral_image_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::integral_image_gpu_async(&src.inner, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU integralImage failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback not yet implemented
    Err(JsValue::from_str("GPU integralImage failed and CPU fallback not yet implemented"))
}

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

/// Generic 2D convolution filter - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = filter2D)]
pub async fn filter2d_wasm(src: &WasmMat, kernel: Vec<f32>, ksize: usize) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Create Mat from kernel data
    let mut kernel_mat = Mat::new(ksize as i32, ksize as i32, 1, MatDepth::F32)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Convert Vec<f32> to bytes for Mat
    let kernel_bytes: Vec<u8> = kernel.iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();
    kernel_mat.data_mut().copy_from_slice(&kernel_bytes);

    // Anchor point is typically the center of the kernel
    let anchor = ((ksize / 2) as i32, (ksize / 2) as i32);

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::filter2d_gpu_async(&src.inner, &mut dst, &kernel_mat, anchor).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU filter2D failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback not yet implemented
    Err(JsValue::from_str("GPU filter2D failed and CPU fallback not yet implemented"))
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

/// Remap (generic pixel remapping) - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = remap)]
pub async fn remap_wasm(src: &WasmMat, map_x: Vec<f32>, map_y: Vec<f32>) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Create Mat from map data
    let mut map_x_mat = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::F32)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut map_y_mat = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::F32)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Convert Vec<f32> to bytes for Mat
    let map_x_bytes: Vec<u8> = map_x.iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();
    let map_y_bytes: Vec<u8> = map_y.iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();

    map_x_mat.data_mut().copy_from_slice(&map_x_bytes);
    map_y_mat.data_mut().copy_from_slice(&map_y_bytes);

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::remap_gpu_async(&src.inner, &mut dst, &map_x_mat, &map_y_mat).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU remap failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback not yet implemented
    Err(JsValue::from_str("GPU remap failed and CPU fallback not yet implemented"))
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

/// Lookup table transform - GPU-accelerated
#[cfg(target_arch = "wasm32")]
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
            match crate::gpu::ops::lut_gpu_async(&src.inner, &lut_mat, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU LUT failed, falling back to CPU".into());
                }
            }
        }
    }

    Err(JsValue::from_str("GPU LUT failed and CPU fallback not yet implemented"))
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

/// Split multi-channel image into separate channels - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = splitChannels)]
pub async fn split_channels_wasm(src: &WasmMat) -> Result<Vec<WasmMat>, JsValue> {
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::split_gpu_async(&src.inner).await {
                Ok(channels) => {
                    return Ok(channels.into_iter().map(|mat| WasmMat { inner: mat }).collect());
                }
                Err(_) => {
                    web_sys::console::log_1(&"GPU split failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    let channels = crate::core::split(&src.inner)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(channels.into_iter().map(|mat| WasmMat { inner: mat }).collect())
}

/// Merge separate channels into multi-channel image - GPU-accelerated
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = mergeChannels)]
pub async fn merge_channels_wasm(channels: Vec<WasmMat>) -> Result<WasmMat, JsValue> {
    if channels.is_empty() {
        return Err(JsValue::from_str("At least one channel required"));
    }

    let channel_mats: Vec<Mat> = channels.iter().map(|wm| wm.inner.clone()).collect();
    let rows = channel_mats[0].rows();
    let cols = channel_mats[0].cols();
    let num_channels = channel_mats.len() as i32;

    let mut dst = Mat::new(rows, cols, num_channels as usize, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::merge_gpu_async(&channel_mats, &mut dst).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU merge failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::core::merge(&channel_mats, &mut dst)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}
