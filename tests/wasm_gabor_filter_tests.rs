//! WASM tests for gabor_filter
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, gabor_filter_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_gabor_filter_smoke() {
    let src = create_test_image_gray();
    assert!(gabor_filter_wasm(&src, 5, 1.0, 0.0, 10.0, 0.5).await.is_ok());
}

#[wasm_bindgen_test]
async fn test_gabor_filter_dimensions() {
    let src = create_test_image_gray();
    let result = gabor_filter_wasm(&src, 5, 1.0, 0.0, 10.0, 0.5).await.unwrap();
    assert!(result.width() > 0 && result.height() > 0);
}

#[wasm_bindgen_test]
async fn test_gabor_filter_backends() {
    let src = create_test_image_gray();
    set_backend_wasm("cpu");
    let cpu = gabor_filter_wasm(&src, 5, 1.0, 0.0, 10.0, 0.5).await;
    set_backend_wasm("auto");
    if cpu.is_ok() {
        let auto_res = gabor_filter_wasm(&src, 5, 1.0, 0.0, 10.0, 0.5).await;
        assert!(auto_res.is_ok());
    }
    set_backend_wasm("auto");
}
