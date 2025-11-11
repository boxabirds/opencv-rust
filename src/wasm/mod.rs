//! WASM bindings for opencv-rust
//!
//! This module provides JavaScript-compatible bindings for running opencv-rust
//! in the browser via WebAssembly.

pub mod backend;
pub mod macros;
pub mod basic;
pub mod imgproc;
pub mod features;
pub mod arithmetic;
pub mod comparison;
pub mod video;
pub mod calib3d;
pub mod dnn;
pub mod ml;
pub mod segmentation;
pub mod misc;

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

// Re-export functions from modules
#[cfg(target_arch = "wasm32")]
pub use basic::threshold::{threshold_wasm, adaptive_threshold_wasm};
#[cfg(target_arch = "wasm32")]
pub use basic::edge::{canny_wasm, sobel_wasm, scharr_wasm, laplacian_wasm};
#[cfg(target_arch = "wasm32")]
pub use basic::filtering::{
    gaussian_blur_wasm, blur_wasm, box_blur_wasm, median_blur_wasm,
    bilateral_filter_wasm, guided_filter_wasm, gabor_filter_wasm
};
#[cfg(target_arch = "wasm32")]
pub use imgproc::morphology::{
    erode_wasm, dilate_wasm, morphology_opening_wasm, morphology_closing_wasm,
    morphology_gradient_wasm, morphology_top_hat_wasm, morphology_black_hat_wasm,
    morphology_tophat_wasm, morphology_blackhat_wasm
};
#[cfg(target_arch = "wasm32")]
pub use imgproc::drawing::{
    draw_line_wasm, draw_rectangle_wasm, draw_circle_wasm,
    draw_ellipse_wasm, draw_polylines_wasm, put_text_wasm
};
#[cfg(target_arch = "wasm32")]
pub use imgproc::geometric::{
    resize_wasm, flip_wasm, rotate_wasm, warp_affine_wasm,
    warp_perspective_wasm, get_rotation_matrix_2d_wasm, remap_wasm
};



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

