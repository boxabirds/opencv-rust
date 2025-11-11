//! WASM tests for guided_filter
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, guided_filter_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_guided_filter_smoke() {
    let src = create_test_image_gray();
    assert!(guided_filter_wasm(&src, 3, 0.1).await.is_ok());
}

#[wasm_bindgen_test]
async fn test_guided_filter_dimensions() {
    let src = create_test_image_gray();
    let result = guided_filter_wasm(&src, 3, 0.1).await.unwrap();
    assert!(result.width() > 0 && result.height() > 0);
}

#[wasm_bindgen_test]
async fn test_guided_filter_backends() {
    let src = create_test_image_gray();
    set_backend_wasm("cpu");
    let cpu = guided_filter_wasm(&src, 3, 0.1).await;
    set_backend_wasm("auto");
    if cpu.is_ok() {
        let auto_res = guided_filter_wasm(&src, 3, 0.1).await;
        assert!(auto_res.is_ok());
    }
    set_backend_wasm("auto");
}
