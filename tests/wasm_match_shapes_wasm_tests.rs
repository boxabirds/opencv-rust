//! WASM tests for match_shapes_wasm
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, match_shapes_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_match_shapes_wasm_smoke() {
    let src = create_test_image_gray();
    let result = match_shapes_wasm(&src, 1.0).await;
    // Test completes without panic
    let _ = result;
}

#[wasm_bindgen_test]
async fn test_match_shapes_wasm_basic() {
    let src = create_test_image_gray();
    if let Ok(result) = match_shapes_wasm(&src, 1.0).await {
        assert!(result.width() > 0 || result.height() > 0);
    }
}

#[wasm_bindgen_test]
async fn test_match_shapes_wasm_cpu() {
    set_backend_wasm("cpu");
    let src = create_test_image_gray();
    let _ = match_shapes_wasm(&src, 1.0).await;
    set_backend_wasm("auto");
}
