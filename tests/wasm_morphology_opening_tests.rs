//! WASM tests for morphology_opening
#![cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, morphology_opening_wasm, set_backend_wasm};
mod wasm_test_utils;
use wasm_test_utils::*;
wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_morphology_opening_smoke() {
    let src = create_test_image_gray();
    assert!(morphology_opening_wasm(&src, 3).await.is_ok());
}

#[wasm_bindgen_test]
async fn test_morphology_opening_dimensions() {
    let src = create_test_image_gray();
    let result = morphology_opening_wasm(&src, 3).await.unwrap();
    assert!(result.width() > 0 && result.height() > 0);
}

#[wasm_bindgen_test]
async fn test_morphology_opening_backends() {
    let src = create_test_image_gray();
    set_backend_wasm("cpu");
    let cpu = morphology_opening_wasm(&src, 3).await;
    set_backend_wasm("auto");
    if cpu.is_ok() {
        let auto_res = morphology_opening_wasm(&src, 3).await;
        assert!(auto_res.is_ok());
    }
    set_backend_wasm("auto");
}
