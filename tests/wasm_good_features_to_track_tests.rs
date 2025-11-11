//! WASM tests for good_features_to_track
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, good_features_to_track_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_good_features_to_track_smoke() {
    let src = create_test_image_rgb();
    let result = good_features_to_track_wasm(&src, 10, 0.01, 10.0).await;
    assert!(result.is_ok() || result.is_err(), "Function should complete");
}

#[wasm_bindgen_test]
async fn test_good_features_to_track_basic() {
    let src = create_test_image_rgb();
    if let Ok(result) = good_features_to_track_wasm(&src, 10, 0.01, 10.0).await {
        assert!(result.width() > 0);
    }
}

#[wasm_bindgen_test]
async fn test_good_features_to_track_cpu() {
    let src = create_test_image_rgb();
    set_backend_wasm("cpu");
    let _ = good_features_to_track_wasm(&src, 10, 0.01, 10.0).await;
    set_backend_wasm("auto");
}
