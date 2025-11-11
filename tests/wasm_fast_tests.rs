//! WASM tests for fast
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, fast_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_fast_smoke() {
    let src = create_test_image_gray();
    assert!(fast_wasm(&src, 20).await.is_ok());
}

#[wasm_bindgen_test]
async fn test_fast_dimensions() {
    let src = create_test_image_gray();
    let result = fast_wasm(&src, 20).await.unwrap();
    assert!(result.width() > 0 && result.height() > 0);
}

#[wasm_bindgen_test]
async fn test_fast_backends() {
    let src = create_test_image_gray();
    set_backend_wasm("cpu");
    let cpu = fast_wasm(&src, 20).await;
    set_backend_wasm("auto");
    if cpu.is_ok() {
        let auto_res = fast_wasm(&src, 20).await;
        assert!(auto_res.is_ok());
    }
    set_backend_wasm("auto");
}
