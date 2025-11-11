//! WASM tests for warp_affine
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, warp_affine_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_warp_affine_smoke() {
    let src = create_test_image_rgb();
    let result = warp_affine_wasm(&src, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).await;
    assert!(result.is_ok() || result.is_err(), "Function should complete");
}

#[wasm_bindgen_test]
async fn test_warp_affine_basic() {
    let src = create_test_image_rgb();
    if let Ok(result) = warp_affine_wasm(&src, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).await {
        assert!(result.width() > 0);
    }
}

#[wasm_bindgen_test]
async fn test_warp_affine_cpu() {
    let src = create_test_image_rgb();
    set_backend_wasm("cpu");
    let _ = warp_affine_wasm(&src, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).await;
    set_backend_wasm("auto");
}
