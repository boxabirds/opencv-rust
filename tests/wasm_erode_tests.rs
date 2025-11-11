//! WASM tests for erode operation
//!
//! Tests OpenCV.js API parity for the erode function

#![cfg(all(target_arch = "wasm32", feature = "wasm"))]

use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, erode_wasm, set_backend_wasm};

mod wasm_test_utils;
use wasm_test_utils::*;

wasm_bindgen_test_configure!(run_in_browser);

// ====================
// 1. SMOKE TESTS
// ====================

#[wasm_bindgen_test]
async fn test_erode_basic_smoke() {
    let src = create_test_image_gray();
    let result = erode_wasm(&src, 3).await;
    assert!(result.is_ok(), "erode_wasm should not fail on valid input");
}

// ====================
// 2. DIMENSION TESTS
// ====================

#[wasm_bindgen_test]
async fn test_erode_output_dimensions() {
    let src = create_test_image_rgb();
    let result = erode_wasm(&src, 3).await.unwrap();

    assert_eq!(result.width(), src.width(), "Width should be preserved");
    assert_eq!(result.height(), src.height(), "Height should be preserved");
    assert_eq!(result.channels(), src.channels(), "Channels should be preserved");
}

#[wasm_bindgen_test]
async fn test_erode_preserves_dimensions() {
    let src = create_test_image_gray();
    let result = erode_wasm(&src, 5).await.unwrap();

    assert!(check_dimensions(&result, src.width(), src.height(), src.channels()));
}

// ====================
// 3. CORRECTNESS TESTS
// ====================

#[wasm_bindgen_test]
async fn test_erode_reduces_white_regions() {
    // Erode should reduce white (foreground) regions
    // Create image with white region
    let width = 10;
    let height = 10;
    let channels = 1;
    let mut data = vec![255u8; width * height * channels]; // All white

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let result = erode_wasm(&src, 3).await.unwrap();

    // After erosion, there should be fewer white pixels (border erosion)
    let src_white = count_nonzero(&src);
    let result_white = count_nonzero(&result);

    assert!(
        result_white <= src_white,
        "Erode should reduce white regions: {} <= {}",
        result_white,
        src_white
    );
}

#[wasm_bindgen_test]
async fn test_erode_preserves_black_background() {
    // Black regions should remain black or expand
    let width = 10;
    let height = 10;
    let channels = 1;
    let data = vec![0u8; width * height * channels]; // All black

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let result = erode_wasm(&src, 3).await.unwrap();

    // Should remain all black
    assert!(is_black(&result), "Black image should remain black after erosion");
}

#[wasm_bindgen_test]
async fn test_erode_removes_small_features() {
    // Erode removes small white features
    let width = 10;
    let height = 10;
    let channels = 1;
    let mut data = vec![0u8; width * height * channels];

    // Add small white feature in center
    data[45] = 255; // Single white pixel at position 45

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let result = erode_wasm(&src, 3).await.unwrap();

    // Small feature should be eroded away
    let result_white = count_nonzero(&result);
    assert!(
        result_white == 0,
        "Small features should be removed by erosion"
    );
}

// ====================
// 4. EDGE CASES
// ====================

#[wasm_bindgen_test]
async fn test_erode_small_image() {
    let width = 5;
    let height = 5;
    let channels = 1;
    let data = vec![128u8; width * height * channels];
    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();

    let result = erode_wasm(&src, 3).await;
    assert!(result.is_ok(), "Should handle small images");
}

#[wasm_bindgen_test]
async fn test_erode_large_image() {
    let src = create_test_image_large();
    let result = erode_wasm(&src, 3).await;

    assert!(result.is_ok(), "Should handle large images");
}

#[wasm_bindgen_test]
async fn test_erode_grayscale() {
    let src = create_test_image_gray();
    let result = erode_wasm(&src, 3).await.unwrap();

    assert_eq!(result.channels(), 1, "Should preserve grayscale");
}

#[wasm_bindgen_test]
async fn test_erode_rgb() {
    let src = create_test_image_rgb();
    let result = erode_wasm(&src, 3).await.unwrap();

    assert_eq!(result.channels(), 3, "Should preserve RGB");
}

// ====================
// 5. PARAMETER TESTS
// ====================

#[wasm_bindgen_test]
async fn test_erode_kernel_size_3() {
    let src = create_test_image_gray();
    let result = erode_wasm(&src, 3).await;
    assert!(result.is_ok(), "Should handle kernel size 3");
}

#[wasm_bindgen_test]
async fn test_erode_kernel_size_5() {
    let src = create_test_image_gray();
    let result = erode_wasm(&src, 5).await;
    assert!(result.is_ok(), "Should handle kernel size 5");
}

#[wasm_bindgen_test]
async fn test_erode_kernel_size_7() {
    let src = create_test_image_gray();
    let result = erode_wasm(&src, 7).await;
    assert!(result.is_ok(), "Should handle kernel size 7");
}

#[wasm_bindgen_test]
async fn test_erode_larger_kernel_more_erosion() {
    // Create image with white region
    let width = 20;
    let height = 20;
    let channels = 1;
    let data = vec![255u8; width * height * channels];
    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();

    let erode3 = erode_wasm(&src, 3).await.unwrap();
    let erode5 = erode_wasm(&src, 5).await.unwrap();
    let erode7 = erode_wasm(&src, 7).await.unwrap();

    let white3 = count_nonzero(&erode3);
    let white5 = count_nonzero(&erode5);
    let white7 = count_nonzero(&erode7);

    // Larger kernel should erode more (fewer white pixels)
    assert!(
        white7 <= white5 && white5 <= white3,
        "Larger kernel should erode more: {} <= {} <= {}",
        white7,
        white5,
        white3
    );
}

// ====================
// 6. BACKEND TESTS
// ====================

#[wasm_bindgen_test]
async fn test_erode_cpu_backend() {
    set_backend_wasm("cpu");

    let src = create_test_image_gray();
    let result = erode_wasm(&src, 3).await;

    assert!(result.is_ok(), "erode_wasm should work with CPU backend");

    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_erode_gpu_backend() {
    set_backend_wasm("webgpu");

    let src = create_test_image_gray();
    let result = erode_wasm(&src, 3).await;

    if let Ok(output) = result {
        assert_eq!(output.width(), src.width());
        assert_eq!(output.height(), src.height());
    }

    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_erode_cpu_gpu_consistency() {
    let src = create_test_image_gray();

    set_backend_wasm("cpu");
    let cpu_result = erode_wasm(&src, 3).await.unwrap();

    set_backend_wasm("auto");
    let auto_result = erode_wasm(&src, 3).await.unwrap();

    // Erode should produce identical results
    assert!(
        images_are_similar(&cpu_result, &auto_result, 1.0),
        "CPU and GPU erode results should be identical"
    );

    set_backend_wasm("auto");
}

// ====================
// 7. OPENCV.JS PARITY
// ====================

#[wasm_bindgen_test]
async fn test_erode_opencv_js_parity() {
    // OpenCV.js cv.erode(src, dst, kernel)
    // Our API: erode_wasm(src, ksize)
    //
    // Reference: https://docs.opencv.org/4.x/d4/d76/tutorial_js_morphological_ops.html

    let src = create_test_image_gray();
    let result = erode_wasm(&src, 3).await.unwrap();

    // Verify dimensions preserved (OpenCV.js behavior)
    assert_eq!(result.width(), src.width());
    assert_eq!(result.height(), src.height());
    assert_eq!(result.channels(), src.channels());
}

// ====================
// 8. CUSTOM TESTS
// ====================

#[wasm_bindgen_test]
async fn test_erode_binary_image() {
    // Binary image with white square
    let width = 10;
    let height = 10;
    let channels = 1;
    let mut data = vec![0u8; width * height * channels];

    // Create white square in center (4x4)
    for y in 3..7 {
        for x in 3..7 {
            data[y * width + x] = 255;
        }
    }

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let src_white = count_nonzero(&src);

    let eroded = erode_wasm(&src, 3).await.unwrap();
    let eroded_white = count_nonzero(&eroded);

    // Square should be eroded (fewer white pixels)
    assert!(
        eroded_white < src_white,
        "Erosion should reduce white square"
    );
}

#[wasm_bindgen_test]
async fn test_erode_repeated() {
    // Applying erode multiple times should progressively erode
    let width = 20;
    let height = 20;
    let channels = 1;
    let data = vec![255u8; width * height * channels];
    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();

    let erode1 = erode_wasm(&src, 3).await.unwrap();
    let erode2 = erode_wasm(&erode1, 3).await.unwrap();
    let erode3 = erode_wasm(&erode2, 3).await.unwrap();

    let white1 = count_nonzero(&erode1);
    let white2 = count_nonzero(&erode2);
    let white3 = count_nonzero(&erode3);

    // Progressive erosion
    assert!(
        white3 <= white2 && white2 <= white1,
        "Repeated erosion should progressively reduce white: {} <= {} <= {}",
        white3,
        white2,
        white1
    );
}

#[wasm_bindgen_test]
async fn test_erode_idempotency_limit() {
    // After enough erosions, image should be completely black
    let width = 10;
    let height = 10;
    let channels = 1;
    let data = vec![255u8; width * height * channels];
    let mut current = WasmMat::from_image_data(&data, width, height, channels).unwrap();

    // Apply erosion many times
    for _ in 0..10 {
        current = erode_wasm(&current, 3).await.unwrap();
    }

    // Should be mostly or completely black
    let remaining_white = count_nonzero(&current);
    let total_pixels = current.width() * current.height();

    assert!(
        remaining_white < total_pixels / 10,
        "After many erosions, most pixels should be black"
    );
}

#[wasm_bindgen_test]
async fn test_erode_thin_line_removal() {
    // Thin lines should be completely removed
    let width = 10;
    let height = 10;
    let channels = 1;
    let mut data = vec![0u8; width * height * channels];

    // Create thin horizontal line
    for x in 0..width {
        data[5 * width + x] = 255; // Row 5
    }

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let eroded = erode_wasm(&src, 3).await.unwrap();

    // Thin line should be removed
    let remaining = count_nonzero(&eroded);
    assert!(
        remaining < 10,
        "Thin line should be mostly removed (remaining={})",
        remaining
    );
}
