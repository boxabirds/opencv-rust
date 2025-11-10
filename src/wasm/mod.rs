//! WASM bindings for opencv-rust
//!
//! This module provides JavaScript-compatible bindings for running opencv-rust
//! in the browser via WebAssembly.

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

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::gaussian_blur_gpu_async(
                &src.inner,
                &mut dst,
                Size::new(ksize as i32, ksize as i32),
                sigma,
            ).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU blur failed, falling back to CPU".into());
                }
            }
        }
    }

    // CPU fallback
    crate::imgproc::gaussian_blur(
        &src.inner,
        &mut dst,
        Size::new(ksize as i32, ksize as i32),
        sigma,
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
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

    // Try GPU first if available
    #[cfg(feature = "gpu")]
    {
        if crate::gpu::gpu_available() {
            match crate::gpu::ops::threshold_gpu_async(
                &src.inner,
                &mut dst,
                thresh as u8,
                max_val as u8,
            ).await {
                Ok(_) => return Ok(WasmMat { inner: dst }),
                Err(_) => {
                    web_sys::console::log_1(&"GPU threshold failed, falling back to CPU".into());
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

    crate::imgproc::blur(
        &src.inner,
        &mut dst,
        Size::new(ksize as i32, ksize as i32),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Median blur (WASM-compatible)
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

    crate::imgproc::median_blur(&src.inner, &mut dst, ksize as i32)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Bilateral filter (WASM-compatible)
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

    crate::imgproc::sobel(&gray, &mut dst, dx, dy, ksize)
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

    crate::imgproc::scharr(&gray, &mut dst, dx, dy)
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

    crate::imgproc::laplacian(&gray, &mut dst, ksize)
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

    crate::imgproc::flip(&src.inner, &mut dst, flip_code)
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

    crate::imgproc::rotate(&src.inner, &mut dst, rotate_enum)
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

    crate::imgproc::cvt_color(&src.inner, &mut dst, code)
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

    crate::imgproc::adaptive_threshold(
        &gray,
        &mut dst,
        maxval,
        AdaptiveThresholdMethod::Mean,
        ThresholdType::Binary,
        block_size,
        c,
    )
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

    let mut dst = Mat::new(height, width, src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    warp_affine(&src.inner, &mut dst, &m, Size::new(width as i32, height as i32))
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
    use crate::imgproc::morphology::{erode, get_structuring_element, MorphShape};
    use crate::core::types::Size;

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(ksize, ksize));
    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        src.inner.depth(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    erode(&src.inner, &mut dst, &kernel)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Morphological dilation
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = dilate)]
pub async fn dilate_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
    use crate::imgproc::morphology::{dilate, get_structuring_element, MorphShape};
    use crate::core::types::Size;

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(ksize, ksize));
    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        src.inner.depth(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    dilate(&src.inner, &mut dst, &kernel)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Morphological opening
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = morphologyOpening)]
pub async fn morphology_opening_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
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

    morphology_ex(&src.inner, &mut dst, MorphType::Open, &kernel)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Morphological closing
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = morphologyClosing)]
pub async fn morphology_closing_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
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

    morphology_ex(&src.inner, &mut dst, MorphType::Close, &kernel)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    Ok(WasmMat { inner: dst })
}

/// Morphological gradient
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = morphologyGradient)]
pub async fn morphology_gradient_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
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

    morphology_ex(&src.inner, &mut dst, MorphType::Gradient, &kernel)
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
    use crate::imgproc::color::cvt_color;
    use crate::core::types::ColorConversionCode;

    let mut dst = Mat::new(
        src.inner.rows(),
        src.inner.cols(),
        src.inner.channels(),
        src.inner.depth(),
    )
    .map_err(|e| JsValue::from_str(&e.to_string()))?;

    cvt_color(&src.inner, &mut dst, ColorConversionCode::BgrToHsv)
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
