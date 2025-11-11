//! WASM tests for canny operation
//!
//! Tests OpenCV.js API parity for the canny edge detector

#![cfg(all(target_arch = "wasm32", feature = "wasm"))]

use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, canny_wasm, set_backend_wasm};

mod wasm_test_utils;
use wasm_test_utils::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_canny_basic_smoke() {
    let src = create_test_image_gray();
    let result = canny_wasm(&src, 100.0, 200.0).await;
    assert!(result.is_ok(), "canny_wasm should not fail on valid input");
}

#[wasm_bindgen_test]
async fn test_canny_output_dimensions() {
    let src = create_test_image_rgb();
    let result = canny_wasm(&src, 100.0, 200.0).await.unwrap();

    assert_eq!(result.width(), src.width());
    assert_eq!(result.height(), src.height());
    assert_eq!(result.channels(), 1, "Canny outputs binary edge map");
}

#[wasm_bindgen_test]
async fn test_canny_detects_edges() {
    let src = create_test_image_rgb();
    let result = canny_wasm(&src, 50.0, 150.0).await.unwrap();

    assert!(count_nonzero(&result) > 0, "Should detect some edges");
    assert!(!is_black(&result), "Should not be completely black");
}

#[wasm_bindgen_test]
async fn test_canny_threshold_parameters() {
    let src = create_test_image_gray();

    let low_thresh = canny_wasm(&src, 30.0, 90.0).await.unwrap();
    let high_thresh = canny_wasm(&src, 100.0, 200.0).await.unwrap();

    let low_edges = count_nonzero(&low_thresh);
    let high_edges = count_nonzero(&high_thresh);

    assert!(
        low_edges >= high_edges,
        "Lower thresholds should detect more edges"
    );
}

#[wasm_bindgen_test]
async fn test_canny_cpu_backend() {
    set_backend_wasm("cpu");
    let src = create_test_image_gray();
    let result = canny_wasm(&src, 100.0, 200.0).await;
    assert!(result.is_ok());
    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_canny_gpu_backend() {
    set_backend_wasm("webgpu");
    let src = create_test_image_gray();
    let result = canny_wasm(&src, 100.0, 200.0).await;
    if let Ok(output) = result {
        assert_eq!(output.channels(), 1);
    }
    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_canny_cpu_gpu_consistency() {
    let src = create_test_image_gray();

    set_backend_wasm("cpu");
    let cpu_result = canny_wasm(&src, 100.0, 200.0).await.unwrap();

    set_backend_wasm("auto");
    let auto_result = canny_wasm(&src, 100.0, 200.0).await.unwrap();

    assert!(images_are_similar(&cpu_result, &auto_result, 10.0));

    set_backend_wasm("auto");
}
