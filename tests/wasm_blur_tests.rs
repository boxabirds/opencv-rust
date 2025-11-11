//! WASM tests for blur operation
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]

use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, blur_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_blur_smoke() {
    assert!(blur_wasm(&create_test_image_rgb(), 5).await.is_ok());
}

#[wasm_bindgen_test]
async fn test_blur_dimensions() {
    let src = create_test_image_rgb();
    let result = blur_wasm(&src, 5).await.unwrap();
    assert!(check_dimensions(&result, src.width(), src.height(), src.channels()));
}

#[wasm_bindgen_test]
async fn test_blur_reduces_stddev() {
    let src = create_test_image_rgb();
    let blurred = blur_wasm(&src, 5).await.unwrap();
    assert!(pixel_stddev(&blurred) < pixel_stddev(&src));
}

#[wasm_bindgen_test]
async fn test_blur_backends() {
    let src = create_test_image_rgb();
    set_backend_wasm("cpu");
    let cpu = blur_wasm(&src, 5).await.unwrap();
    set_backend_wasm("auto");
    let auto_res = blur_wasm(&src, 5).await.unwrap();
    assert!(images_are_similar(&cpu, &auto_res, 5.0));
    set_backend_wasm("auto");
}
