//! WASM tests for median_blur
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, median_blur_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_median_blur_smoke() {
    assert!(median_blur_wasm(&create_test_image_rgb(), 5).await.is_ok());
}

#[wasm_bindgen_test]
async fn test_median_blur_dimensions() {
    let src = create_test_image_rgb();
    let result = median_blur_wasm(&src, 5).await.unwrap();
    assert_eq!(result.width(), src.width());
    assert_eq!(result.height(), src.height());
}

#[wasm_bindgen_test]
async fn test_median_blur_backends() {
    let src = create_test_image_rgb();
    set_backend_wasm("cpu");
    let cpu = median_blur_wasm(&src, 5).await.unwrap();
    set_backend_wasm("auto");
    let auto_res = median_blur_wasm(&src, 5).await.unwrap();
    assert!(images_are_similar(&cpu, &auto_res, 5.0));
    set_backend_wasm("auto");
}
