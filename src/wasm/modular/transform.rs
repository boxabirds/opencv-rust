//! Transform operations for WASM

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::*;
#[cfg(target_arch = "wasm32")]
use crate::wasm::{WasmMat, backend};


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
