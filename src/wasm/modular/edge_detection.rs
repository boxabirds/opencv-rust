//! Edge Detection operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::*;
#[cfg(target_arch = "wasm32")]
use crate::wasm::{WasmMat, backend};


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
