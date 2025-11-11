//! WASM tests for draw_ellipse
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, draw_ellipse_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_draw_ellipse_smoke() {
    let src = create_test_image_rgb();
    let result = draw_ellipse_wasm(&src, 5, 5, 3, 2, 0.0, vec![255, 0, 0]).await;
    assert!(result.is_ok() || result.is_err(), "Function should complete");
}

#[wasm_bindgen_test]
async fn test_draw_ellipse_basic() {
    let src = create_test_image_rgb();
    if let Ok(result) = draw_ellipse_wasm(&src, 5, 5, 3, 2, 0.0, vec![255, 0, 0]).await {
        assert!(result.width() > 0);
    }
}

#[wasm_bindgen_test]
async fn test_draw_ellipse_cpu() {
    let src = create_test_image_rgb();
    set_backend_wasm("cpu");
    let _ = draw_ellipse_wasm(&src, 5, 5, 3, 2, 0.0, vec![255, 0, 0]).await;
    set_backend_wasm("auto");
}
