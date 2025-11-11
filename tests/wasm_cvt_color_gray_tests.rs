//! WASM tests for cvt_color_gray
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, cvt_color_gray_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_cvt_color_gray_smoke() {
    assert!(cvt_color_gray_wasm(&create_test_image_rgb()).await.is_ok());
}

#[wasm_bindgen_test]
async fn test_cvt_color_gray_dimensions() {
    let src = create_test_image_rgb();
    let result = cvt_color_gray_wasm(&src).await.unwrap();
    assert_eq!(result.width(), src.width());
    assert_eq!(result.height(), src.height());
}

#[wasm_bindgen_test]
async fn test_cvt_color_gray_backends() {
    let src = create_test_image_rgb();
    set_backend_wasm("cpu");
    let cpu = cvt_color_gray_wasm(&src).await.unwrap();
    set_backend_wasm("auto");
    let auto_res = cvt_color_gray_wasm(&src).await.unwrap();
    assert!(images_are_similar(&cpu, &auto_res, 5.0));
    set_backend_wasm("auto");
}
