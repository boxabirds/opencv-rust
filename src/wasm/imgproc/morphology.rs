//! Morphological operations

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use crate::core::{Mat, MatDepth};
#[cfg(target_arch = "wasm32")]
use crate::wasm::WasmMat;
#[cfg(target_arch = "wasm32")]
use crate::wasm::backend;

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

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::erode_gpu_async(&src.inner, &mut dst, ksize)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            erode(&src.inner, &mut dst, &kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

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

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::dilate_gpu_async(&src.inner, &mut dst, ksize)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            dilate(&src.inner, &mut dst, &kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Morphological opening - GPU-accelerated
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

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::morphology_opening_gpu_async(&src.inner, &mut dst, ksize)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            morphology_ex(&src.inner, &mut dst, MorphType::Open, &kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Morphological closing - GPU-accelerated
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

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::morphology_closing_gpu_async(&src.inner, &mut dst, ksize)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            morphology_ex(&src.inner, &mut dst, MorphType::Close, &kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Morphological gradient - GPU-accelerated
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

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::morphology_gradient_gpu_async(&src.inner, &mut dst, ksize)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            morphology_ex(&src.inner, &mut dst, MorphType::Gradient, &kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

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

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::morphology_tophat_gpu_async(&src.inner, &mut dst, ksize)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            morphology_ex(&src.inner, &mut dst, MorphType::TopHat, &kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

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

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::morphology_blackhat_gpu_async(&src.inner, &mut dst, ksize)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            morphology_ex(&src.inner, &mut dst, MorphType::BlackHat, &kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Morphological top hat (alternative casing)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = morphologyTophat)]
pub async fn morphology_tophat_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
    use crate::imgproc::morphology::{morphology_ex, get_structuring_element, MorphShape, MorphType};
    use crate::core::types::Size;

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(ksize, ksize));
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::morphology_tophat_gpu_async(&src.inner, &mut dst, ksize)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            morphology_ex(&src.inner, &mut dst, MorphType::TopHat, &kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}

/// Morphological black hat (alternative casing)
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = morphologyBlackhat)]
pub async fn morphology_blackhat_wasm(src: &WasmMat, ksize: i32) -> Result<WasmMat, JsValue> {
    use crate::imgproc::morphology::{morphology_ex, get_structuring_element, MorphShape, MorphType};
    use crate::core::types::Size;

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(ksize, ksize));
    let mut dst = Mat::new(src.inner.rows(), src.inner.cols(), src.inner.channels(), src.inner.depth())
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Backend dispatch
    crate::backend_dispatch! {
        gpu => {
            crate::gpu::ops::morphology_blackhat_gpu_async(&src.inner, &mut dst, ksize)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
        cpu => {
            morphology_ex(&src.inner, &mut dst, MorphType::BlackHat, &kernel)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}
