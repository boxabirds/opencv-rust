//! WASM tests for [OPERATION_NAME] operation
//!
//! Tests OpenCV.js API parity for the [OPERATION_NAME] function
//!
//! USAGE: Copy this template and replace:
//! - [OPERATION_NAME] → operation name (e.g., "resize", "sobel")
//! - [operation_wasm] → actual function name
//! - [PARAMS] → function parameters
//! - Add operation-specific tests

#![cfg(all(target_arch = "wasm32", feature = "wasm"))]

use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, [operation_wasm], set_backend_wasm};

mod wasm_test_utils;
use wasm_test_utils::*;

wasm_bindgen_test_configure!(run_in_browser);

// ====================
// 1. SMOKE TESTS
// ====================

#[wasm_bindgen_test]
async fn test_[operation]_basic_smoke() {
    // Basic smoke test - function doesn't panic
    let src = create_test_image_rgb();
    let result = [operation_wasm](&src, [PARAMS]).await;
    assert!(result.is_ok(), "[operation_wasm] should not fail on valid input");
}

// ====================
// 2. DIMENSION TESTS
// ====================

#[wasm_bindgen_test]
async fn test_[operation]_output_dimensions() {
    // Verify output dimensions are correct
    let src = create_test_image_rgb();
    let result = [operation_wasm](&src, [PARAMS]).await.unwrap();

    // Adjust assertions based on operation behavior:
    // - Preserve dimensions: assert_eq!(result.width(), src.width())
    // - Transform dimensions: assert_eq!(result.width(), expected_width)
    assert_eq!(result.width(), src.width(), "Width should be [preserved/transformed]");
    assert_eq!(result.height(), src.height(), "Height should be [preserved/transformed]");
    assert_eq!(result.channels(), src.channels(), "Channels should be [preserved/transformed]");
}

// ====================
// 3. CORRECTNESS TESTS
// ====================

#[wasm_bindgen_test]
async fn test_[operation]_correctness() {
    // Test that operation produces expected output
    let src = create_test_image_gray();
    let result = [operation_wasm](&src, [PARAMS]).await.unwrap();

    // Add operation-specific validation:
    // - For filters: check smoothness, stddev, etc.
    // - For transforms: check pixel values, boundaries
    // - For detectors: check feature counts, positions

    // Example assertions:
    // assert!(pixel_stddev(&result) < pixel_stddev(&src), "Should smooth image");
    // assert!(!is_black(&result), "Should produce non-trivial output");
    // assert!(count_nonzero(&result) > 0, "Should have non-zero pixels");
}

// ====================
// 4. EDGE CASES
// ====================

#[wasm_bindgen_test]
async fn test_[operation]_small_image() {
    // Test with minimal size image
    let width = 3;
    let height = 3;
    let channels = 1;
    let data = vec![128u8; width * height * channels];
    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();

    let result = [operation_wasm](&src, [PARAMS]).await;
    assert!(result.is_ok(), "Should handle small images");
}

#[wasm_bindgen_test]
async fn test_[operation]_large_image() {
    // Test with larger image
    let src = create_test_image_large();
    let result = [operation_wasm](&src, [PARAMS]).await;

    assert!(result.is_ok(), "Should handle large images");
}

#[wasm_bindgen_test]
async fn test_[operation]_grayscale() {
    // Test with grayscale input
    let src = create_test_image_gray();
    let result = [operation_wasm](&src, [PARAMS]).await;

    assert!(result.is_ok(), "Should handle grayscale images");
}

#[wasm_bindgen_test]
async fn test_[operation]_rgb() {
    // Test with RGB input
    let src = create_test_image_rgb();
    let result = [operation_wasm](&src, [PARAMS]).await;

    assert!(result.is_ok(), "Should handle RGB images");
}

// ====================
// 5. PARAMETER TESTS
// ====================

#[wasm_bindgen_test]
async fn test_[operation]_parameter_variations() {
    // Test different parameter values
    let src = create_test_image_rgb();

    // Test parameter set 1
    let result1 = [operation_wasm](&src, [PARAMS_1]).await;
    assert!(result1.is_ok(), "Should handle parameter set 1");

    // Test parameter set 2
    let result2 = [operation_wasm](&src, [PARAMS_2]).await;
    assert!(result2.is_ok(), "Should handle parameter set 2");

    // Test parameter set 3
    let result3 = [operation_wasm](&src, [PARAMS_3]).await;
    assert!(result3.is_ok(), "Should handle parameter set 3");

    // Optional: Compare results to verify parameter effects
    // Example: assert!(pixel_stddev(&result1) != pixel_stddev(&result2));
}

// ====================
// 6. BACKEND TESTS
// ====================

#[wasm_bindgen_test]
async fn test_[operation]_cpu_backend() {
    // Test with explicit CPU backend
    set_backend_wasm("cpu");

    let src = create_test_image_rgb();
    let result = [operation_wasm](&src, [PARAMS]).await;

    assert!(result.is_ok(), "[operation_wasm] should work with CPU backend");

    // Reset to auto
    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_[operation]_gpu_backend() {
    // Test with explicit WebGPU backend (may fail if GPU not available)
    set_backend_wasm("webgpu");

    let src = create_test_image_rgb();
    let result = [operation_wasm](&src, [PARAMS]).await;

    // Result can be Ok or Err depending on GPU availability
    if let Ok(output) = result {
        assert_eq!(output.width(), src.width());
        assert_eq!(output.height(), src.height());
    }

    // Reset to auto
    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_[operation]_cpu_gpu_consistency() {
    // CPU and GPU should produce similar results
    let src = create_test_image_rgb();

    // Test with CPU
    set_backend_wasm("cpu");
    let cpu_result = [operation_wasm](&src, [PARAMS]).await.unwrap();

    // Test with auto (may use GPU)
    set_backend_wasm("auto");
    let auto_result = [operation_wasm](&src, [PARAMS]).await.unwrap();

    // Results should be similar (adjust tolerance based on operation)
    // For deterministic operations: tolerance = 0.0
    // For floating-point operations: tolerance = 1.0-5.0
    assert!(
        images_are_similar(&cpu_result, &auto_result, 5.0),
        "CPU and GPU results should be similar"
    );

    // Reset to auto
    set_backend_wasm("auto");
}

// ====================
// 7. OPENCV.JS PARITY
// ====================

#[wasm_bindgen_test]
async fn test_[operation]_opencv_js_parity() {
    // Test specific OpenCV.js behaviors
    // Reference: https://docs.opencv.org/4.x/d5/d0f/tutorial_js_table_of_contents_imgproc.html

    // Example: Verify operation handles edge cases like OpenCV.js
    // Example: Verify operation returns expected type/shape like OpenCV.js
    // Example: Verify operation parameters match OpenCV.js semantics

    let src = create_test_image_rgb();
    let result = [operation_wasm](&src, [PARAMS]).await.unwrap();

    // Add parity checks here based on OpenCV.js documentation
    assert!(check_dimensions(&result, src.width(), src.height(), src.channels()));
}

// ====================
// 8. CUSTOM TESTS
// ====================
// Add operation-specific tests below

// For example, if testing a transform operation:
// #[wasm_bindgen_test]
// async fn test_[operation]_preserves_property() { ... }

// For example, if testing a detector:
// #[wasm_bindgen_test]
// async fn test_[operation]_detects_features() { ... }

// For example, if testing a morphological operation:
// #[wasm_bindgen_test]
// async fn test_[operation]_affects_structure() { ... }
