//! WASM tests for bilateral_filter
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, bilateral_filter_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_bilateral_filter_smoke() {
    assert!(bilateral_filter_wasm(&create_test_image_rgb(), 5, 50.0, 50.0).await.is_ok());
}

#[wasm_bindgen_test]
async fn test_bilateral_filter_dimensions() {
    let src = create_test_image_rgb();
    let result = bilateral_filter_wasm(&src, 5, 50.0, 50.0).await.unwrap();
    assert_eq!(result.width(), src.width());
    assert_eq!(result.height(), src.height());
}

#[wasm_bindgen_test]
async fn test_bilateral_filter_backends() {
    let src = create_test_image_rgb();
    set_backend_wasm("cpu");
    let cpu = bilateral_filter_wasm(&src, 5, 50.0, 50.0).await.unwrap();
    set_backend_wasm("auto");
    let auto_res = bilateral_filter_wasm(&src, 5, 50.0, 50.0).await.unwrap();
    assert!(images_are_similar(&cpu, &auto_res, 5.0));
    set_backend_wasm("auto");
}
