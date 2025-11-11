//! WASM tests for equalize_histogram
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, equalize_histogram_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_equalize_histogram_smoke() {
    let src = create_test_image_gray();
    assert!(equalize_histogram_wasm(&src).await.is_ok());
}

#[wasm_bindgen_test]
async fn test_equalize_histogram_dimensions() {
    let src = create_test_image_gray();
    let result = equalize_histogram_wasm(&src).await.unwrap();
    assert!(result.width() > 0 && result.height() > 0);
}

#[wasm_bindgen_test]
async fn test_equalize_histogram_backends() {
    let src = create_test_image_gray();
    set_backend_wasm("cpu");
    let cpu = equalize_histogram_wasm(&src).await;
    set_backend_wasm("auto");
    if cpu.is_ok() {
        let auto_res = equalize_histogram_wasm(&src).await;
        assert!(auto_res.is_ok());
    }
    set_backend_wasm("auto");
}
