//! WASM tests for calc_histogram
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, calc_histogram_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_calc_histogram_smoke() {
    let src = create_test_image_gray();
    assert!(calc_histogram_wasm(&src, 256).await.is_ok());
}

#[wasm_bindgen_test]
async fn test_calc_histogram_dimensions() {
    let src = create_test_image_gray();
    let result = calc_histogram_wasm(&src, 256).await.unwrap();
    assert!(result.width() > 0 && result.height() > 0);
}

#[wasm_bindgen_test]
async fn test_calc_histogram_backends() {
    let src = create_test_image_gray();
    set_backend_wasm("cpu");
    let cpu = calc_histogram_wasm(&src, 256).await;
    set_backend_wasm("auto");
    if cpu.is_ok() {
        let auto_res = calc_histogram_wasm(&src, 256).await;
        assert!(auto_res.is_ok());
    }
    set_backend_wasm("auto");
}
