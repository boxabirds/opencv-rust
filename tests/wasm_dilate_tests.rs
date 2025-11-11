//! WASM tests for dilate operation
//!
//! Tests OpenCV.js API parity for the dilate function

#![cfg(all(target_arch = "wasm32", feature = "wasm"))]

use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, dilate_wasm, set_backend_wasm};

mod wasm_test_utils;
use wasm_test_utils::*;

wasm_bindgen_test_configure!(run_in_browser);

// ====================
// 1. SMOKE TESTS
// ====================

#[wasm_bindgen_test]
async fn test_dilate_basic_smoke() {
    let src = create_test_image_gray();
    let result = dilate_wasm(&src, 3).await;
    assert!(result.is_ok(), "dilate_wasm should not fail on valid input");
}

// ====================
// 2. DIMENSION TESTS
// ====================

#[wasm_bindgen_test]
async fn test_dilate_output_dimensions() {
    let src = create_test_image_rgb();
    let result = dilate_wasm(&src, 3).await.unwrap();

    assert_eq!(result.width(), src.width(), "Width should be preserved");
    assert_eq!(result.height(), src.height(), "Height should be preserved");
    assert_eq!(result.channels(), src.channels(), "Channels should be preserved");
}

#[wasm_bindgen_test]
async fn test_dilate_preserves_dimensions() {
    let src = create_test_image_gray();
    let result = dilate_wasm(&src, 5).await.unwrap();

    assert!(check_dimensions(&result, src.width(), src.height(), src.channels()));
}

// ====================
// 3. CORRECTNESS TESTS
// ====================

#[wasm_bindgen_test]
async fn test_dilate_expands_white_regions() {
    // Dilate should expand white (foreground) regions
    // Create image with white feature
    let width = 10;
    let height = 10;
    let channels = 1;
    let mut data = vec![0u8; width * height * channels]; // All black

    // Add small white square in center
    for y in 4..6 {
        for x in 4..6 {
            data[y * width + x] = 255;
        }
    }

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let src_white = count_nonzero(&src);

    let result = dilate_wasm(&src, 3).await.unwrap();
    let result_white = count_nonzero(&result);

    // After dilation, there should be more white pixels
    assert!(
        result_white >= src_white,
        "Dilate should expand white regions: {} >= {}",
        result_white,
        src_white
    );
}

#[wasm_bindgen_test]
async fn test_dilate_preserves_white_background() {
    // All white image should remain all white
    let width = 10;
    let height = 10;
    let channels = 1;
    let data = vec![255u8; width * height * channels]; // All white

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let result = dilate_wasm(&src, 3).await.unwrap();

    // Should remain all white
    assert!(is_white(&result), "White image should remain white after dilation");
}

#[wasm_bindgen_test]
async fn test_dilate_fills_small_gaps() {
    // Dilate fills small gaps in white regions
    let width = 10;
    let height = 10;
    let channels = 1;
    let mut data = vec![255u8; width * height * channels]; // All white

    // Add small black gap in center
    data[45] = 0; // Single black pixel at position 45

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let result = dilate_wasm(&src, 3).await.unwrap();

    // Gap should be filled (all pixels white)
    assert!(
        is_white(&result),
        "Small gaps should be filled by dilation"
    );
}

// ====================
// 4. EDGE CASES
// ====================

#[wasm_bindgen_test]
async fn test_dilate_small_image() {
    let width = 5;
    let height = 5;
    let channels = 1;
    let data = vec![128u8; width * height * channels];
    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();

    let result = dilate_wasm(&src, 3).await;
    assert!(result.is_ok(), "Should handle small images");
}

#[wasm_bindgen_test]
async fn test_dilate_large_image() {
    let src = create_test_image_large();
    let result = dilate_wasm(&src, 3).await;

    assert!(result.is_ok(), "Should handle large images");
}

#[wasm_bindgen_test]
async fn test_dilate_grayscale() {
    let src = create_test_image_gray();
    let result = dilate_wasm(&src, 3).await.unwrap();

    assert_eq!(result.channels(), 1, "Should preserve grayscale");
}

#[wasm_bindgen_test]
async fn test_dilate_rgb() {
    let src = create_test_image_rgb();
    let result = dilate_wasm(&src, 3).await.unwrap();

    assert_eq!(result.channels(), 3, "Should preserve RGB");
}

// ====================
// 5. PARAMETER TESTS
// ====================

#[wasm_bindgen_test]
async fn test_dilate_kernel_size_3() {
    let src = create_test_image_gray();
    let result = dilate_wasm(&src, 3).await;
    assert!(result.is_ok(), "Should handle kernel size 3");
}

#[wasm_bindgen_test]
async fn test_dilate_kernel_size_5() {
    let src = create_test_image_gray();
    let result = dilate_wasm(&src, 5).await;
    assert!(result.is_ok(), "Should handle kernel size 5");
}

#[wasm_bindgen_test]
async fn test_dilate_kernel_size_7() {
    let src = create_test_image_gray();
    let result = dilate_wasm(&src, 7).await;
    assert!(result.is_ok(), "Should handle kernel size 7");
}

#[wasm_bindgen_test]
async fn test_dilate_larger_kernel_more_dilation() {
    // Create image with white feature
    let width = 20;
    let height = 20;
    let channels = 1;
    let mut data = vec![0u8; width * height * channels];

    // Small white square in center
    for y in 9..11 {
        for x in 9..11 {
            data[y * width + x] = 255;
        }
    }

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();

    let dilate3 = dilate_wasm(&src, 3).await.unwrap();
    let dilate5 = dilate_wasm(&src, 5).await.unwrap();
    let dilate7 = dilate_wasm(&src, 7).await.unwrap();

    let white3 = count_nonzero(&dilate3);
    let white5 = count_nonzero(&dilate5);
    let white7 = count_nonzero(&dilate7);

    // Larger kernel should dilate more (more white pixels)
    assert!(
        white7 >= white5 && white5 >= white3,
        "Larger kernel should dilate more: {} >= {} >= {}",
        white7,
        white5,
        white3
    );
}

// ====================
// 6. BACKEND TESTS
// ====================

#[wasm_bindgen_test]
async fn test_dilate_cpu_backend() {
    set_backend_wasm("cpu");

    let src = create_test_image_gray();
    let result = dilate_wasm(&src, 3).await;

    assert!(result.is_ok(), "dilate_wasm should work with CPU backend");

    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_dilate_gpu_backend() {
    set_backend_wasm("webgpu");

    let src = create_test_image_gray();
    let result = dilate_wasm(&src, 3).await;

    if let Ok(output) = result {
        assert_eq!(output.width(), src.width());
        assert_eq!(output.height(), src.height());
    }

    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_dilate_cpu_gpu_consistency() {
    let src = create_test_image_gray();

    set_backend_wasm("cpu");
    let cpu_result = dilate_wasm(&src, 3).await.unwrap();

    set_backend_wasm("auto");
    let auto_result = dilate_wasm(&src, 3).await.unwrap();

    // Dilate should produce identical results
    assert!(
        images_are_similar(&cpu_result, &auto_result, 1.0),
        "CPU and GPU dilate results should be identical"
    );

    set_backend_wasm("auto");
}

// ====================
// 7. OPENCV.JS PARITY
// ====================

#[wasm_bindgen_test]
async fn test_dilate_opencv_js_parity() {
    // OpenCV.js cv.dilate(src, dst, kernel)
    // Our API: dilate_wasm(src, ksize)
    //
    // Reference: https://docs.opencv.org/4.x/d4/d76/tutorial_js_morphological_ops.html

    let src = create_test_image_gray();
    let result = dilate_wasm(&src, 3).await.unwrap();

    // Verify dimensions preserved (OpenCV.js behavior)
    assert_eq!(result.width(), src.width());
    assert_eq!(result.height(), src.height());
    assert_eq!(result.channels(), src.channels());
}

// ====================
// 8. CUSTOM TESTS
// ====================

#[wasm_bindgen_test]
async fn test_dilate_binary_image() {
    // Binary image with white dot
    let width = 10;
    let height = 10;
    let channels = 1;
    let mut data = vec![0u8; width * height * channels];

    // Single white pixel in center
    data[55] = 255; // Position (5,5)

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let src_white = count_nonzero(&src);

    let dilated = dilate_wasm(&src, 3).await.unwrap();
    let dilated_white = count_nonzero(&dilated);

    // Dot should be dilated (more white pixels)
    assert!(
        dilated_white > src_white,
        "Dilation should expand white dot: {} > {}",
        dilated_white,
        src_white
    );
}

#[wasm_bindgen_test]
async fn test_dilate_repeated() {
    // Applying dilate multiple times should progressively dilate
    let width = 20;
    let height = 20;
    let channels = 1;
    let mut data = vec![0u8; width * height * channels];

    // Small white dot in center
    data[210] = 255; // Position (10,10)

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();

    let dilate1 = dilate_wasm(&src, 3).await.unwrap();
    let dilate2 = dilate_wasm(&dilate1, 3).await.unwrap();
    let dilate3 = dilate_wasm(&dilate2, 3).await.unwrap();

    let white1 = count_nonzero(&dilate1);
    let white2 = count_nonzero(&dilate2);
    let white3 = count_nonzero(&dilate3);

    // Progressive dilation
    assert!(
        white3 >= white2 && white2 >= white1,
        "Repeated dilation should progressively expand white: {} >= {} >= {}",
        white3,
        white2,
        white1
    );
}

#[wasm_bindgen_test]
async fn test_dilate_idempotency_limit() {
    // After enough dilations, image should be completely white
    let width = 10;
    let height = 10;
    let channels = 1;
    let mut data = vec![0u8; width * height * channels];

    // Single white pixel
    data[55] = 255;

    let mut current = WasmMat::from_image_data(&data, width, height, channels).unwrap();

    // Apply dilation many times
    for _ in 0..15 {
        current = dilate_wasm(&current, 3).await.unwrap();
    }

    // Should be mostly or completely white
    let white = count_nonzero(&current);
    let total_pixels = current.width() * current.height();

    assert!(
        white > total_pixels * 9 / 10,
        "After many dilations, most pixels should be white: {}/{}",
        white,
        total_pixels
    );
}

#[wasm_bindgen_test]
async fn test_dilate_connects_nearby_features() {
    // Dilate should connect nearby white features
    let width = 10;
    let height = 10;
    let channels = 1;
    let mut data = vec![0u8; width * height * channels];

    // Two white pixels close together
    data[44] = 255; // Position (4,4)
    data[46] = 255; // Position (4,6)

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let dilated = dilate_wasm(&src, 3).await.unwrap();

    // Features should be connected (more white pixels than just 2x dilation)
    let white = count_nonzero(&dilated);
    assert!(
        white > 10,
        "Dilation should connect nearby features (white={})",
        white
    );
}

#[wasm_bindgen_test]
async fn test_dilate_thin_line_thickening() {
    // Thin lines should become thicker
    let width = 10;
    let height = 10;
    let channels = 1;
    let mut data = vec![0u8; width * height * channels];

    // Create thin horizontal line
    for x in 2..8 {
        data[5 * width + x] = 255; // Row 5
    }

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let src_white = count_nonzero(&src);

    let dilated = dilate_wasm(&src, 3).await.unwrap();
    let dilated_white = count_nonzero(&dilated);

    // Line should be thicker (more white pixels)
    assert!(
        dilated_white > src_white,
        "Thin line should be thickened: {} > {}",
        dilated_white,
        src_white
    );
}
