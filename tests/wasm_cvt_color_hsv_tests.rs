//! WASM tests for cvt_color_hsv
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, cvt_color_hsv_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_cvt_color_hsv_smoke() {
    let src = create_test_image_gray();
    assert!(cvt_color_hsv_wasm(&src).await.is_ok());
}

#[wasm_bindgen_test]
async fn test_cvt_color_hsv_dimensions() {
    let src = create_test_image_gray();
    let result = cvt_color_hsv_wasm(&src).await.unwrap();
    assert!(result.width() > 0 && result.height() > 0);
}

#[wasm_bindgen_test]
async fn test_cvt_color_hsv_backends() {
    let src = create_test_image_gray();
    set_backend_wasm("cpu");
    let cpu = cvt_color_hsv_wasm(&src).await;
    set_backend_wasm("auto");
    if cpu.is_ok() {
        let auto_res = cvt_color_hsv_wasm(&src).await;
        assert!(auto_res.is_ok());
    }
    set_backend_wasm("auto");
}
