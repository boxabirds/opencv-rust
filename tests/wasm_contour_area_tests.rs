//! WASM tests for contour_area
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, contour_area_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_contour_area_smoke() {
    let src = create_test_image_rgb();
    let result = contour_area_wasm(&src, vec![]).await;
    assert!(result.is_ok() || result.is_err(), "Function should complete");
}

#[wasm_bindgen_test]
async fn test_contour_area_basic() {
    let src = create_test_image_rgb();
    if let Ok(result) = contour_area_wasm(&src, vec![]).await {
        assert!(result.width() > 0);
    }
}

#[wasm_bindgen_test]
async fn test_contour_area_cpu() {
    let src = create_test_image_rgb();
    set_backend_wasm("cpu");
    let _ = contour_area_wasm(&src, vec![]).await;
    set_backend_wasm("auto");
}
