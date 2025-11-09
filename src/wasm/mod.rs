//! WASM bindings for opencv-rust
//!
//! This module provides JavaScript-compatible bindings for running opencv-rust
//! in the browser via WebAssembly.

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::Size;
#[cfg(target_arch = "wasm32")]
use crate::error::Result;

/// Initialize the WASM module with panic hooks for better error messages
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_init() {
    console_error_panic_hook::set_once();
    web_sys::console::log_1(&"OpenCV-Rust WASM initialized".into());
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
}

/// Gaussian blur operation (WASM-compatible)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = gaussianBlur)]
pub fn gaussian_blur_wasm(
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

    crate::imgproc::gaussian_blur(
        &src.inner,
        &mut dst,
        Size::new(ksize as i32, ksize as i32),
        sigma,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Resize operation (WASM-compatible)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = resize)]
pub fn resize_wasm(
    src: &WasmMat,
    dst_width: usize,
    dst_height: usize,
) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(dst_height, dst_width, src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::imgproc::resize(
        &src.inner,
        &mut dst,
        Size::new(dst_width as i32, dst_height as i32),
        crate::imgproc::InterpolationMethod::Linear,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Threshold operation (WASM-compatible)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = threshold)]
pub fn threshold_wasm(
    src: &WasmMat,
    thresh: f64,
    max_val: f64,
) -> Result<WasmMat, JsValue> {
    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut gray = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        crate::imgproc::cvt_color(
            &src.inner,
            &mut gray,
            crate::imgproc::ColorConversion::BGR2GRAY,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        gray
    } else {
        src.inner.clone()
    };

    let mut dst = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::imgproc::threshold(
        &gray,
        &mut dst,
        thresh,
        max_val,
        crate::imgproc::ThresholdType::Binary,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Canny edge detection (WASM-compatible)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = canny)]
pub fn canny_wasm(
    src: &WasmMat,
    threshold1: f64,
    threshold2: f64,
) -> Result<WasmMat, JsValue> {
    // Convert to grayscale if needed
    let gray = if src.inner.channels() > 1 {
        let mut gray = Mat::new(src.inner.rows(), src.inner.cols(), 1, MatDepth::U8)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        crate::imgproc::cvt_color(
            &src.inner,
            &mut gray,
            crate::imgproc::ColorConversion::BGR2GRAY,
        )
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
        gray
    } else {
        src.inner.clone()
    };

    let mut dst = Mat::new(gray.rows(), gray.cols(), 1, MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    crate::imgproc::canny(&gray, &mut dst, threshold1, threshold2, 3)
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
