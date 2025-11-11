//! Geometric transformations

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::core::types::{Size, InterpolationFlag};
#[cfg(target_arch = "wasm32")]
use crate::wasm::WasmMat;
#[cfg(target_arch = "wasm32")]
use crate::wasm::backend;

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

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::resize_gpu_async(&src.inner, &mut dst, dst_width, dst_height)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::resize(
                &src.inner,
                &mut dst,
                Size::new(dst_width as i32, dst_height as i32),
                InterpolationFlag::Linear,
            )
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

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

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::flip_gpu_async(&src.inner, &mut dst, flip_code)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::flip(&src.inner, &mut dst, flip_code)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

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

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::rotate_gpu_async(&src.inner, &mut dst, rotate_code)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            crate::imgproc::rotate(&src.inner, &mut dst, rotate_enum)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

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

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            let m_gpu: [f32; 6] = [
                matrix[0] as f32, matrix[1] as f32, matrix[2] as f32,
                matrix[3] as f32, matrix[4] as f32, matrix[5] as f32,
            ];
            crate::gpu::ops::warp_affine_gpu_async(&src.inner, &mut dst, &m_gpu, (width, height))
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            warp_affine(&src.inner, &mut dst, &m, Size::new(width as i32, height as i32))
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

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

    let transform_matrix = [
        [m11, m12, m13],
        [m21, m22, m23],
        [m31, m32, m33],
    ];

    let dsize = Size::new(src.inner.cols() as i32, src.inner.rows() as i32);
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            let m_gpu: [f32; 9] = [
                m11 as f32, m12 as f32, m13 as f32,
                m21 as f32, m22 as f32, m23 as f32,
                m31 as f32, m32 as f32, m33 as f32,
            ];
            crate::gpu::ops::warp_perspective_gpu_async(&src.inner, &mut dst, &m_gpu, (src.inner.cols(), src.inner.rows()))
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            warp_perspective(&src.inner, &mut dst, &transform_matrix, dsize)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

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
    use crate::core::types::Point2f;

    let center = Point2f::new(center_x as f32, center_y as f32);
    let rotation_matrix = get_rotation_matrix_2d(center, angle, scale);

    let dsize = Size::new(src.inner.cols() as i32, src.inner.rows() as i32);
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            let m_gpu: [f32; 6] = [
                rotation_matrix[0][0] as f32, rotation_matrix[0][1] as f32, rotation_matrix[0][2] as f32,
                rotation_matrix[1][0] as f32, rotation_matrix[1][1] as f32, rotation_matrix[1][2] as f32,
            ];
            crate::gpu::ops::warp_affine_gpu_async(&src.inner, &mut dst, &m_gpu, (src.inner.cols(), src.inner.rows()))
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            warp_affine(&src.inner, &mut dst, &rotation_matrix, dsize)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

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

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::remap_gpu_async(&src.inner, &mut dst, &map_x_mat, &map_y_mat)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            return Err(JsValue::from_str("CPU remap not yet implemented"));
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== pyrDown =====
#[wasm_bindgen(js_name = pyrDown)]
pub async fn pyr_down_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let dst_width = src.inner.cols() / 2;
    let dst_height = src.inner.rows() / 2;
    let mut dst = Mat::new(dst_height, dst_width, src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::pyrdown_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            return Err(JsValue::from_str("CPU pyrDown not yet implemented"));
        }
    }

    Ok(WasmMat { inner: dst })
}


// ===== pyrUp =====
#[wasm_bindgen(js_name = pyrUp)]
pub async fn pyr_up_wasm(src: &WasmMat) -> Result<WasmMat, JsValue> {
    let dst_width = src.inner.cols() * 2;
    let dst_height = src.inner.rows() * 2;
    let mut dst = Mat::new(dst_height, dst_width, src.inner.channels(), MatDepth::U8)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::pyrup_gpu_async(&src.inner, &mut dst)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            return Err(JsValue::from_str("CPU pyrUp not yet implemented"));
        }
    }

    Ok(WasmMat { inner: dst })
}


