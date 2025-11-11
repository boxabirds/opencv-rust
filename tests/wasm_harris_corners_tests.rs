//! WASM tests for harris_corners
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, harris_corners_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_harris_corners_smoke() {
    let src = create_test_image_gray();
    assert!(harris_corners_wasm(&src, 2, 3, 0.04).await.is_ok());
}

#[wasm_bindgen_test]
async fn test_harris_corners_dimensions() {
    let src = create_test_image_gray();
    let result = harris_corners_wasm(&src, 2, 3, 0.04).await.unwrap();
    assert!(result.width() > 0 && result.height() > 0);
}

#[wasm_bindgen_test]
async fn test_harris_corners_backends() {
    let src = create_test_image_gray();
    set_backend_wasm("cpu");
    let cpu = harris_corners_wasm(&src, 2, 3, 0.04).await;
    set_backend_wasm("auto");
    if cpu.is_ok() {
        let auto_res = harris_corners_wasm(&src, 2, 3, 0.04).await;
        assert!(auto_res.is_ok());
    }
    set_backend_wasm("auto");
}
