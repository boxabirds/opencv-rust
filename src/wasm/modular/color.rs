//! Color conversion operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::ColorConversionCode;
#[cfg(target_arch = "wasm32")]
use crate::wasm::WasmMat;


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
