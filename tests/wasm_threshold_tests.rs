//! WASM tests for threshold operation
//!
//! Tests OpenCV.js API parity for the threshold function

#![cfg(all(target_arch = "wasm32", feature = "wasm"))]

use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, threshold_wasm, set_backend_wasm};

mod wasm_test_utils;
use wasm_test_utils::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_threshold_basic_smoke() {
    // Basic smoke test - function doesn't panic
    let src = create_test_image_gray();
    let result = threshold_wasm(&src, 128.0, 255.0).await;
    assert!(result.is_ok(), "threshold_wasm should not fail on valid input");
}

#[wasm_bindgen_test]
async fn test_threshold_output_dimensions() {
    // Output should match input dimensions
    let src = create_test_image_gray();
    let result = threshold_wasm(&src, 128.0, 255.0).await.unwrap();

    assert_eq!(result.width(), src.width(), "Width should be preserved");
    assert_eq!(result.height(), src.height(), "Height should be preserved");
    // Note: threshold converts to grayscale, so channels should be 1
    assert_eq!(result.channels(), 1, "Threshold output should be grayscale");
}

#[wasm_bindgen_test]
async fn test_threshold_binary_behavior() {
    // Test that threshold produces binary output (only 0 or max_val)
    let src = create_test_image_gray();
    let thresh = 128.0;
    let max_val = 255.0;

    let result = threshold_wasm(&src, thresh, max_val).await.unwrap();
    let data = result.get_data();

    // All pixels should be either 0 or max_val
    for &pixel in data.iter() {
        assert!(
            pixel == 0 || pixel == max_val as u8,
            "Pixel value {} should be either 0 or {}",
            pixel,
            max_val
        );
    }
}

#[wasm_bindgen_test]
async fn test_threshold_all_below() {
    // Image with all pixels below threshold → all zeros
    let width = 10;
    let height = 10;
    let channels = 1;
    let data = vec![50u8; width * height * channels]; // All pixels = 50

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let result = threshold_wasm(&src, 100.0, 255.0).await.unwrap();

    assert!(is_black(&result), "All pixels below threshold should result in black image");
}

#[wasm_bindgen_test]
async fn test_threshold_all_above() {
    // Image with all pixels above threshold → all max_val
    let width = 10;
    let height = 10;
    let channels = 1;
    let data = vec![200u8; width * height * channels]; // All pixels = 200

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let result = threshold_wasm(&src, 100.0, 255.0).await.unwrap();

    assert!(is_white(&result), "All pixels above threshold should result in white image");
}

#[wasm_bindgen_test]
async fn test_threshold_exact_boundary() {
    // Test pixels exactly at threshold value
    let width = 10;
    let height = 10;
    let channels = 1;
    let thresh_val = 128.0;
    let data = vec![thresh_val as u8; width * height * channels];

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let result = threshold_wasm(&src, thresh_val, 255.0).await.unwrap();
    let output_data = result.get_data();

    // Pixels at exact threshold should be thresholded consistently
    // (OpenCV behavior: pixel > thresh → max_val, else → 0)
    // So pixel == thresh should be 0
    for &pixel in output_data.iter() {
        assert_eq!(pixel, 0, "Pixel at exact threshold should be 0");
    }
}

#[wasm_bindgen_test]
async fn test_threshold_different_max_values() {
    // Test with different max_val values
    let src = create_test_image_gray();

    let result_255 = threshold_wasm(&src, 128.0, 255.0).await.unwrap();
    let result_200 = threshold_wasm(&src, 128.0, 200.0).await.unwrap();
    let result_100 = threshold_wasm(&src, 128.0, 100.0).await.unwrap();

    let data_255 = result_255.get_data();
    let data_200 = result_200.get_data();
    let data_100 = result_100.get_data();

    // Check that max values are respected
    let max_255 = data_255.iter().max().unwrap();
    let max_200 = data_200.iter().max().unwrap();
    let max_100 = data_100.iter().max().unwrap();

    assert!(
        *max_255 <= 255,
        "Max value should not exceed specified max_val"
    );
    assert!(
        *max_200 <= 200,
        "Max value should not exceed specified max_val"
    );
    assert!(
        *max_100 <= 100,
        "Max value should not exceed specified max_val"
    );
}

#[wasm_bindgen_test]
async fn test_threshold_rgb_to_gray_conversion() {
    // threshold should handle RGB input by converting to grayscale
    let src = create_test_image_rgb();
    assert_eq!(src.channels(), 3, "Source should be RGB");

    let result = threshold_wasm(&src, 128.0, 255.0).await.unwrap();

    assert_eq!(result.channels(), 1, "Output should be grayscale");
    assert_eq!(result.width(), src.width(), "Width should be preserved");
    assert_eq!(result.height(), src.height(), "Height should be preserved");
}

#[wasm_bindgen_test]
async fn test_threshold_cpu_backend() {
    // Test with explicit CPU backend
    set_backend_wasm("cpu");

    let src = create_test_image_gray();
    let result = threshold_wasm(&src, 128.0, 255.0).await;

    assert!(result.is_ok(), "threshold_wasm should work with CPU backend");

    // Reset to auto
    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_threshold_gpu_backend() {
    // Test with explicit WebGPU backend (may fail if GPU not available)
    set_backend_wasm("webgpu");

    let src = create_test_image_gray();
    let result = threshold_wasm(&src, 128.0, 255.0).await;

    // Result can be Ok or Err depending on GPU availability
    // If Ok, verify output is valid
    if let Ok(output) = result {
        assert_eq!(output.width(), src.width());
        assert_eq!(output.height(), src.height());
    }

    // Reset to auto
    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_threshold_cpu_gpu_consistency() {
    // CPU and GPU should produce similar results
    let src = create_test_image_gray();

    // Test with CPU
    set_backend_wasm("cpu");
    let cpu_result = threshold_wasm(&src, 128.0, 255.0).await.unwrap();

    // Test with GPU (if available)
    set_backend_wasm("auto");
    let auto_result = threshold_wasm(&src, 128.0, 255.0).await.unwrap();

    // Results should be identical for binary threshold
    assert!(
        images_are_similar(&cpu_result, &auto_result, 0.0),
        "CPU and GPU threshold results should be identical"
    );

    // Reset to auto
    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_threshold_large_image() {
    // Test with larger image for performance validation
    let src = create_test_image_large();
    let result = threshold_wasm(&src, 128.0, 255.0).await;

    assert!(result.is_ok(), "threshold_wasm should handle large images");

    let output = result.unwrap();
    assert_eq!(output.width(), src.width());
    assert_eq!(output.height(), src.height());

    // Verify binary output
    let data = output.get_data();
    for &pixel in data.iter() {
        assert!(pixel == 0 || pixel == 255);
    }
}

#[wasm_bindgen_test]
async fn test_threshold_zero_threshold() {
    // Edge case: threshold = 0 (all pixels above threshold)
    let src = create_test_image_gray();
    let result = threshold_wasm(&src, 0.0, 255.0).await.unwrap();

    // With threshold=0, all non-zero pixels should become max_val
    let nonzero_count = count_nonzero(&result);
    assert!(nonzero_count > 0, "Should have non-zero pixels with threshold=0");
}

#[wasm_bindgen_test]
async fn test_threshold_max_threshold() {
    // Edge case: threshold = 255 (all pixels below threshold)
    let src = create_test_image_gray();
    let result = threshold_wasm(&src, 255.0, 255.0).await.unwrap();

    // With threshold=255, all pixels should be 0
    assert!(is_black(&result), "All pixels should be 0 with threshold=255");
}
