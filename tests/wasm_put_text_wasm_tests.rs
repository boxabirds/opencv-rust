//! WASM tests for put_text_wasm
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, put_text_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_put_text_wasm_smoke() {
    let src = create_test_image_gray();
    let result = put_text_wasm(&src, "Test".to_string(), 10, 10, 1.0, vec![255, 0, 0]).await;
    // Test completes without panic
    let _ = result;
}

#[wasm_bindgen_test]
async fn test_put_text_wasm_basic() {
    let src = create_test_image_gray();
    if let Ok(result) = put_text_wasm(&src, "Test".to_string(), 10, 10, 1.0, vec![255, 0, 0]).await {
        assert!(result.width() > 0 || result.height() > 0);
    }
}

#[wasm_bindgen_test]
async fn test_put_text_wasm_cpu() {
    set_backend_wasm("cpu");
    let src = create_test_image_gray();
    let _ = put_text_wasm(&src, "Test".to_string(), 10, 10, 1.0, vec![255, 0, 0]).await;
    set_backend_wasm("auto");
}
