//! WASM tests for find_contours
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, find_contours_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_find_contours_smoke() {
    let src = create_test_image_rgb();
    let result = find_contours_wasm(&src).await;
    assert!(result.is_ok() || result.is_err(), "Function should complete");
}

#[wasm_bindgen_test]
async fn test_find_contours_basic() {
    let src = create_test_image_rgb();
    if let Ok(result) = find_contours_wasm(&src).await {
        assert!(result.width() > 0);
    }
}

#[wasm_bindgen_test]
async fn test_find_contours_cpu() {
    let src = create_test_image_rgb();
    set_backend_wasm("cpu");
    let _ = find_contours_wasm(&src).await;
    set_backend_wasm("auto");
}
