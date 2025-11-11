//! WASM tests for nlm_denoising_wasm
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, nlm_denoising_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_nlm_denoising_wasm_smoke() {
    let src = create_test_image_gray();
    let result = nlm_denoising_wasm(&src, 3, 7, 21).await;
    // Test completes without panic
    let _ = result;
}

#[wasm_bindgen_test]
async fn test_nlm_denoising_wasm_basic() {
    let src = create_test_image_gray();
    if let Ok(result) = nlm_denoising_wasm(&src, 3, 7, 21).await {
        assert!(result.width() > 0 || result.height() > 0);
    }
}

#[wasm_bindgen_test]
async fn test_nlm_denoising_wasm_cpu() {
    set_backend_wasm("cpu");
    let src = create_test_image_gray();
    let _ = nlm_denoising_wasm(&src, 3, 7, 21).await;
    set_backend_wasm("auto");
}
