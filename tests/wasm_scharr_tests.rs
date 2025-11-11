//! WASM tests for scharr
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, scharr_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_scharr_smoke() {
    assert!(scharr_wasm(&create_test_image_rgb(), 1, 0).await.is_ok());
}

#[wasm_bindgen_test]
async fn test_scharr_dimensions() {
    let src = create_test_image_rgb();
    let result = scharr_wasm(&src, 1, 0).await.unwrap();
    assert_eq!(result.width(), src.width());
    assert_eq!(result.height(), src.height());
}

#[wasm_bindgen_test]
async fn test_scharr_backends() {
    let src = create_test_image_rgb();
    set_backend_wasm("cpu");
    let cpu = scharr_wasm(&src, 1, 0).await.unwrap();
    set_backend_wasm("auto");
    let auto_res = scharr_wasm(&src, 1, 0).await.unwrap();
    assert!(images_are_similar(&cpu, &auto_res, 5.0));
    set_backend_wasm("auto");
}
