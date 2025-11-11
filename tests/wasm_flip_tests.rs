//! WASM tests for flip operation
//!
//! Tests OpenCV.js API parity for the flip function

#![cfg(all(target_arch = "wasm32", feature = "wasm"))]

use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, flip_wasm, set_backend_wasm};

mod wasm_test_utils;
use wasm_test_utils::*;

wasm_bindgen_test_configure!(run_in_browser);

// ====================
// 1. SMOKE TESTS
// ====================

#[wasm_bindgen_test]
async fn test_flip_basic_smoke() {
    let src = create_test_image_gray();
    let result = flip_wasm(&src, 0).await;
    assert!(result.is_ok(), "flip_wasm should not fail on valid input");
}

// ====================
// 2. DIMENSION TESTS
// ====================

#[wasm_bindgen_test]
async fn test_flip_output_dimensions() {
    let src = create_test_image_rgb();
    let result = flip_wasm(&src, 0).await.unwrap();

    assert_eq!(result.width(), src.width(), "Width should be preserved");
    assert_eq!(result.height(), src.height(), "Height should be preserved");
    assert_eq!(result.channels(), src.channels(), "Channels should be preserved");
}

#[wasm_bindgen_test]
async fn test_flip_preserves_dimensions() {
    let src = create_test_image_gray();

    let flip_v = flip_wasm(&src, 0).await.unwrap();
    assert!(check_dimensions(&flip_v, src.width(), src.height(), src.channels()));

    let flip_h = flip_wasm(&src, 1).await.unwrap();
    assert!(check_dimensions(&flip_h, src.width(), src.height(), src.channels()));

    let flip_both = flip_wasm(&src, -1).await.unwrap();
    assert!(check_dimensions(&flip_both, src.width(), src.height(), src.channels()));
}

// ====================
// 3. CORRECTNESS TESTS
// ====================

#[wasm_bindgen_test]
async fn test_flip_vertical() {
    // Create asymmetric image (top different from bottom)
    let width = 4;
    let height = 4;
    let channels = 1;
    let mut data = vec![0u8; width * height * channels];

    // Top half white, bottom half black
    for y in 0..2 {
        for x in 0..width {
            data[y * width + x] = 255;
        }
    }

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let flipped = flip_wasm(&src, 0).await.unwrap();
    let flipped_data = flipped.get_data();

    // After vertical flip, bottom should be white, top black
    assert_eq!(flipped_data[0], 0, "Top-left should be black after vertical flip");
    assert_eq!(flipped_data[12], 255, "Bottom-left should be white after vertical flip");
}

#[wasm_bindgen_test]
async fn test_flip_horizontal() {
    // Create asymmetric image (left different from right)
    let width = 4;
    let height = 4;
    let channels = 1;
    let mut data = vec![0u8; width * height * channels];

    // Left half white, right half black
    for y in 0..height {
        for x in 0..2 {
            data[y * width + x] = 255;
        }
    }

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let flipped = flip_wasm(&src, 1).await.unwrap();
    let flipped_data = flipped.get_data();

    // After horizontal flip, right should be white, left black
    assert_eq!(flipped_data[0], 0, "Top-left should be black after horizontal flip");
    assert_eq!(flipped_data[3], 255, "Top-right should be white after horizontal flip");
}

#[wasm_bindgen_test]
async fn test_flip_both() {
    // Flip both is equivalent to 180-degree rotation
    let width = 4;
    let height = 4;
    let channels = 1;
    let mut data = vec![0u8; width * height * channels];

    // Top-left corner white
    data[0] = 255;
    data[1] = 255;
    data[4] = 255;
    data[5] = 255;

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let flipped = flip_wasm(&src, -1).await.unwrap();
    let flipped_data = flipped.get_data();

    // After both flips, bottom-right should be white
    assert_eq!(flipped_data[0], 0, "Top-left should be black");
    assert_eq!(flipped_data[15], 255, "Bottom-right should be white");
}

#[wasm_bindgen_test]
async fn test_flip_double_inversion() {
    // Flipping twice should return to original
    let src = create_test_image_gray();

    let flip_once = flip_wasm(&src, 0).await.unwrap();
    let flip_twice = flip_wasm(&flip_once, 0).await.unwrap();

    // Should be identical to original
    assert!(
        images_are_similar(&src, &flip_twice, 0.0),
        "Flipping twice should return to original"
    );
}

// ====================
// 4. EDGE CASES
// ====================

#[wasm_bindgen_test]
async fn test_flip_small_image() {
    let width = 2;
    let height = 2;
    let channels = 1;
    let data = vec![1u8, 2u8, 3u8, 4u8];
    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();

    let result = flip_wasm(&src, 0).await;
    assert!(result.is_ok(), "Should handle small images");
}

#[wasm_bindgen_test]
async fn test_flip_large_image() {
    let src = create_test_image_large();
    let result = flip_wasm(&src, 0).await;

    assert!(result.is_ok(), "Should handle large images");
}

#[wasm_bindgen_test]
async fn test_flip_grayscale() {
    let src = create_test_image_gray();
    let result = flip_wasm(&src, 0).await.unwrap();

    assert_eq!(result.channels(), 1, "Should preserve grayscale");
}

#[wasm_bindgen_test]
async fn test_flip_rgb() {
    let src = create_test_image_rgb();
    let result = flip_wasm(&src, 0).await.unwrap();

    assert_eq!(result.channels(), 3, "Should preserve RGB");
}

#[wasm_bindgen_test]
async fn test_flip_symmetric_image() {
    // Symmetric image should look the same after flip
    let width = 6;
    let height = 6;
    let channels = 1;
    let mut data = vec![0u8; width * height * channels];

    // Create vertically symmetric pattern
    for y in 0..height {
        for x in 0..width {
            data[y * width + x] = (x * 40) as u8;
        }
    }

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let flipped = flip_wasm(&src, 0).await.unwrap(); // Vertical flip

    // Should be very similar (vertically symmetric)
    assert!(
        images_are_similar(&src, &flipped, 0.0),
        "Vertically symmetric image should look same after vertical flip"
    );
}

// ====================
// 5. PARAMETER TESTS
// ====================

#[wasm_bindgen_test]
async fn test_flip_code_0() {
    // flip_code = 0: vertical flip
    let src = create_test_image_gray();
    let result = flip_wasm(&src, 0).await;
    assert!(result.is_ok(), "Should handle flip_code=0");
}

#[wasm_bindgen_test]
async fn test_flip_code_1() {
    // flip_code = 1: horizontal flip
    let src = create_test_image_gray();
    let result = flip_wasm(&src, 1).await;
    assert!(result.is_ok(), "Should handle flip_code=1");
}

#[wasm_bindgen_test]
async fn test_flip_code_minus_1() {
    // flip_code = -1: both
    let src = create_test_image_gray();
    let result = flip_wasm(&src, -1).await;
    assert!(result.is_ok(), "Should handle flip_code=-1");
}

#[wasm_bindgen_test]
async fn test_flip_different_codes_different_results() {
    let src = create_test_image_gray();

    let flip_v = flip_wasm(&src, 0).await.unwrap();
    let flip_h = flip_wasm(&src, 1).await.unwrap();
    let flip_both = flip_wasm(&src, -1).await.unwrap();

    // All three should be different
    assert!(
        !images_are_similar(&flip_v, &flip_h, 0.0),
        "Vertical and horizontal flip should be different"
    );
    assert!(
        !images_are_similar(&flip_v, &flip_both, 0.0),
        "Vertical flip and both should be different"
    );
    assert!(
        !images_are_similar(&flip_h, &flip_both, 0.0),
        "Horizontal flip and both should be different"
    );
}

// ====================
// 6. BACKEND TESTS
// ====================

#[wasm_bindgen_test]
async fn test_flip_cpu_backend() {
    set_backend_wasm("cpu");

    let src = create_test_image_gray();
    let result = flip_wasm(&src, 0).await;

    assert!(result.is_ok(), "flip_wasm should work with CPU backend");

    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_flip_gpu_backend() {
    set_backend_wasm("webgpu");

    let src = create_test_image_gray();
    let result = flip_wasm(&src, 0).await;

    if let Ok(output) = result {
        assert_eq!(output.width(), src.width());
        assert_eq!(output.height(), src.height());
    }

    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_flip_cpu_gpu_consistency() {
    let src = create_test_image_gray();

    set_backend_wasm("cpu");
    let cpu_result = flip_wasm(&src, 0).await.unwrap();

    set_backend_wasm("auto");
    let auto_result = flip_wasm(&src, 0).await.unwrap();

    // Flip should produce identical results
    assert!(
        images_are_similar(&cpu_result, &auto_result, 0.0),
        "CPU and GPU flip results should be identical"
    );

    set_backend_wasm("auto");
}

// ====================
// 7. OPENCV.JS PARITY
// ====================

#[wasm_bindgen_test]
async fn test_flip_opencv_js_parity() {
    // OpenCV.js cv.flip(src, dst, flipCode)
    // Our API: flip_wasm(src, flip_code)
    //
    // flip_code:
    //   0: flip vertically (around x-axis)
    //   >0: flip horizontally (around y-axis)
    //   <0: flip both
    //
    // Reference: https://docs.opencv.org/4.x/d2/de8/group__core__array.html#gaca7be533e3dac7feb70fc60635adf441

    let src = create_test_image_gray();

    let flip_v = flip_wasm(&src, 0).await.unwrap();
    assert_eq!(flip_v.width(), src.width());
    assert_eq!(flip_v.height(), src.height());

    let flip_h = flip_wasm(&src, 1).await.unwrap();
    assert_eq!(flip_h.width(), src.width());
    assert_eq!(flip_h.height(), src.height());

    let flip_both = flip_wasm(&src, -1).await.unwrap();
    assert_eq!(flip_both.width(), src.width());
    assert_eq!(flip_both.height(), src.height());
}

// ====================
// 8. CUSTOM TESTS
// ====================

#[wasm_bindgen_test]
async fn test_flip_composition() {
    // Horizontal + vertical = both
    let src = create_test_image_gray();

    let flip_h = flip_wasm(&src, 1).await.unwrap();
    let flip_h_v = flip_wasm(&flip_h, 0).await.unwrap();

    let flip_both = flip_wasm(&src, -1).await.unwrap();

    // Should be identical
    assert!(
        images_are_similar(&flip_h_v, &flip_both, 0.0),
        "H+V flip should equal both flip"
    );
}

#[wasm_bindgen_test]
async fn test_flip_preserves_pixel_values() {
    // Flip should preserve all pixel values, just rearrange them
    let src = create_test_image_gray();
    let flipped = flip_wasm(&src, 0).await.unwrap();

    // Average should be identical
    let src_avg = average_pixel_value(&src);
    let flipped_avg = average_pixel_value(&flipped);

    assert_eq!(
        src_avg, flipped_avg,
        "Flip should preserve average pixel value"
    );
}

#[wasm_bindgen_test]
async fn test_flip_corner_pixels() {
    // Create image with distinct corner values
    let width = 4;
    let height = 4;
    let channels = 1;
    let mut data = vec![128u8; width * height * channels];

    // Set corners to distinct values
    data[0] = 10; // Top-left
    data[3] = 20; // Top-right
    data[12] = 30; // Bottom-left
    data[15] = 40; // Bottom-right

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();

    // Vertical flip: swap top and bottom
    let flip_v = flip_wasm(&src, 0).await.unwrap();
    let flip_v_data = flip_v.get_data();
    assert_eq!(flip_v_data[0], 30, "Top-left becomes bottom-left");
    assert_eq!(flip_v_data[12], 10, "Bottom-left becomes top-left");

    // Horizontal flip: swap left and right
    let flip_h = flip_wasm(&src, 1).await.unwrap();
    let flip_h_data = flip_h.get_data();
    assert_eq!(flip_h_data[0], 20, "Top-left becomes top-right");
    assert_eq!(flip_h_data[3], 10, "Top-right becomes top-left");

    // Both flip: swap diagonally opposite
    let flip_both = flip_wasm(&src, -1).await.unwrap();
    let flip_both_data = flip_both.get_data();
    assert_eq!(flip_both_data[0], 40, "Top-left becomes bottom-right");
    assert_eq!(flip_both_data[15], 10, "Bottom-right becomes top-left");
}

#[wasm_bindgen_test]
async fn test_flip_sequence() {
    // Multiple flips in sequence
    let src = create_test_image_gray();

    // H -> V -> H -> V should return to original
    let mut current = flip_wasm(&src, 1).await.unwrap(); // H
    current = flip_wasm(&current, 0).await.unwrap(); // V
    current = flip_wasm(&current, 1).await.unwrap(); // H
    current = flip_wasm(&current, 0).await.unwrap(); // V

    assert!(
        images_are_similar(&src, &current, 0.0),
        "H-V-H-V sequence should return to original"
    );
}
