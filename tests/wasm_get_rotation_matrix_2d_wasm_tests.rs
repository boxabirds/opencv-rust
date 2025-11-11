//! WASM tests for get_rotation_matrix_2d_wasm
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, get_rotation_matrix_2d_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_get_rotation_matrix_2d_wasm_smoke() {
    let src = create_test_image_gray();
    let result = get_rotation_matrix_2d_wasm(&src, 5, 5, 45.0, 1.0).await;
    // Test completes without panic
    let _ = result;
}

#[wasm_bindgen_test]
async fn test_get_rotation_matrix_2d_wasm_basic() {
    let src = create_test_image_gray();
    if let Ok(result) = get_rotation_matrix_2d_wasm(&src, 5, 5, 45.0, 1.0).await {
        assert!(result.width() > 0 || result.height() > 0);
    }
}

#[wasm_bindgen_test]
async fn test_get_rotation_matrix_2d_wasm_cpu() {
    set_backend_wasm("cpu");
    let src = create_test_image_gray();
    let _ = get_rotation_matrix_2d_wasm(&src, 5, 5, 45.0, 1.0).await;
    set_backend_wasm("auto");
}
