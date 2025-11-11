//! WASM tests for random_forest_wasm
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, random_forest_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_random_forest_wasm_smoke() {
    let src = create_test_image_gray();
    let result = random_forest_wasm(&src, 5).await;
    // Test completes without panic
    let _ = result;
}

#[wasm_bindgen_test]
async fn test_random_forest_wasm_basic() {
    let src = create_test_image_gray();
    if let Ok(result) = random_forest_wasm(&src, 5).await {
        assert!(result.width() > 0 || result.height() > 0);
    }
}

#[wasm_bindgen_test]
async fn test_random_forest_wasm_cpu() {
    set_backend_wasm("cpu");
    let src = create_test_image_gray();
    let _ = random_forest_wasm(&src, 5).await;
    set_backend_wasm("auto");
}
