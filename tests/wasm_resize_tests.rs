//! WASM tests for resize operation
//!
//! Tests OpenCV.js API parity for the resize function

#![cfg(all(target_arch = "wasm32", feature = "wasm"))]

use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, resize_wasm, set_backend_wasm};

mod wasm_test_utils;
use wasm_test_utils::*;

wasm_bindgen_test_configure!(run_in_browser);

// ====================
// 1. SMOKE TESTS
// ====================

#[wasm_bindgen_test]
async fn test_resize_basic_smoke() {
    let src = create_test_image_rgb();
    let result = resize_wasm(&src, 20, 20).await;
    assert!(result.is_ok(), "resize_wasm should not fail on valid input");
}

// ====================
// 2. DIMENSION TESTS
// ====================

#[wasm_bindgen_test]
async fn test_resize_output_dimensions() {
    let src = create_test_image_rgb();
    let target_width = 20;
    let target_height = 15;

    let result = resize_wasm(&src, target_width, target_height).await.unwrap();

    assert_eq!(result.width(), target_width, "Width should match target");
    assert_eq!(result.height(), target_height, "Height should match target");
    assert_eq!(result.channels(), src.channels(), "Channels should be preserved");
}

#[wasm_bindgen_test]
async fn test_resize_upscale() {
    let src = create_test_image_rgb(); // 10x10
    let result = resize_wasm(&src, 50, 50).await.unwrap();

    assert_eq!(result.width(), 50);
    assert_eq!(result.height(), 50);
}

#[wasm_bindgen_test]
async fn test_resize_downscale() {
    let src = create_test_image_rgb(); // 10x10
    let result = resize_wasm(&src, 5, 5).await.unwrap();

    assert_eq!(result.width(), 5);
    assert_eq!(result.height(), 5);
}

#[wasm_bindgen_test]
async fn test_resize_non_uniform_scale() {
    let src = create_test_image_rgb(); // 10x10
    let result = resize_wasm(&src, 20, 5).await.unwrap();

    assert_eq!(result.width(), 20, "Width scaled differently");
    assert_eq!(result.height(), 5, "Height scaled differently");
}

// ====================
// 3. CORRECTNESS TESTS
// ====================

#[wasm_bindgen_test]
async fn test_resize_preserves_average_intensity() {
    let src = create_test_image_rgb();
    let src_avg = average_pixel_value(&src);

    let resized = resize_wasm(&src, 20, 20).await.unwrap();
    let resized_avg = average_pixel_value(&resized);

    let diff = (src_avg - resized_avg).abs();
    assert!(
        diff < 15.0,
        "Average intensity should be roughly preserved (diff={})",
        diff
    );
}

#[wasm_bindgen_test]
async fn test_resize_identity() {
    // Resizing to same dimensions should preserve image
    let src = create_test_image_rgb();
    let resized = resize_wasm(&src, src.width(), src.height()).await.unwrap();

    assert!(
        images_are_similar(&src, &resized, 2.0),
        "Resizing to same dimensions should preserve image"
    );
}

// ====================
// 4. EDGE CASES
// ====================

#[wasm_bindgen_test]
async fn test_resize_to_1x1() {
    let src = create_test_image_rgb();
    let result = resize_wasm(&src, 1, 1).await;

    assert!(result.is_ok(), "Should handle resize to 1x1");
    let output = result.unwrap();
    assert_eq!(output.width(), 1);
    assert_eq!(output.height(), 1);
}

#[wasm_bindgen_test]
async fn test_resize_large_upscale() {
    let src = create_test_image_gray(); // 10x10
    let result = resize_wasm(&src, 200, 200).await;

    assert!(result.is_ok(), "Should handle large upscale");
    let output = result.unwrap();
    assert_eq!(output.width(), 200);
    assert_eq!(output.height(), 200);
}

#[wasm_bindgen_test]
async fn test_resize_extreme_aspect_ratio() {
    let src = create_test_image_rgb(); // 10x10
    let result = resize_wasm(&src, 100, 1).await;

    assert!(result.is_ok(), "Should handle extreme aspect ratio");
    let output = result.unwrap();
    assert_eq!(output.width(), 100);
    assert_eq!(output.height(), 1);
}

#[wasm_bindgen_test]
async fn test_resize_grayscale() {
    let src = create_test_image_gray();
    let result = resize_wasm(&src, 20, 20).await.unwrap();

    assert_eq!(result.width(), 20);
    assert_eq!(result.height(), 20);
    assert_eq!(result.channels(), 1, "Should preserve grayscale");
}

#[wasm_bindgen_test]
async fn test_resize_rgb() {
    let src = create_test_image_rgb();
    let result = resize_wasm(&src, 20, 20).await.unwrap();

    assert_eq!(result.width(), 20);
    assert_eq!(result.height(), 20);
    assert_eq!(result.channels(), 3, "Should preserve RGB");
}

// ====================
// 5. PARAMETER TESTS
// ====================

#[wasm_bindgen_test]
async fn test_resize_various_dimensions() {
    let src = create_test_image_rgb();

    // Test various target sizes
    let dimensions = [(5, 5), (10, 10), (15, 20), (25, 10), (50, 50)];

    for (width, height) in dimensions.iter() {
        let result = resize_wasm(&src, *width, *height).await;
        assert!(
            result.is_ok(),
            "Should handle {}x{} resize",
            width,
            height
        );

        let output = result.unwrap();
        assert_eq!(output.width(), *width);
        assert_eq!(output.height(), *height);
    }
}

#[wasm_bindgen_test]
async fn test_resize_scale_factors() {
    let src = create_test_image_rgb(); // 10x10

    // Test 2x scale
    let scale_2x = resize_wasm(&src, 20, 20).await.unwrap();
    assert_eq!(scale_2x.width(), 20);
    assert_eq!(scale_2x.height(), 20);

    // Test 0.5x scale
    let scale_half = resize_wasm(&src, 5, 5).await.unwrap();
    assert_eq!(scale_half.width(), 5);
    assert_eq!(scale_half.height(), 5);

    // Test 4x scale
    let scale_4x = resize_wasm(&src, 40, 40).await.unwrap();
    assert_eq!(scale_4x.width(), 40);
    assert_eq!(scale_4x.height(), 40);
}

// ====================
// 6. BACKEND TESTS
// ====================

#[wasm_bindgen_test]
async fn test_resize_cpu_backend() {
    set_backend_wasm("cpu");

    let src = create_test_image_rgb();
    let result = resize_wasm(&src, 20, 20).await;

    assert!(result.is_ok(), "resize_wasm should work with CPU backend");

    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_resize_gpu_backend() {
    set_backend_wasm("webgpu");

    let src = create_test_image_rgb();
    let result = resize_wasm(&src, 20, 20).await;

    if let Ok(output) = result {
        assert_eq!(output.width(), 20);
        assert_eq!(output.height(), 20);
    }

    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_resize_cpu_gpu_consistency() {
    let src = create_test_image_rgb();

    set_backend_wasm("cpu");
    let cpu_result = resize_wasm(&src, 20, 20).await.unwrap();

    set_backend_wasm("auto");
    let auto_result = resize_wasm(&src, 20, 20).await.unwrap();

    // Resizing may have small differences due to interpolation
    assert!(
        images_are_similar(&cpu_result, &auto_result, 5.0),
        "CPU and GPU resize results should be similar"
    );

    set_backend_wasm("auto");
}

// ====================
// 7. OPENCV.JS PARITY
// ====================

#[wasm_bindgen_test]
async fn test_resize_opencv_js_parity() {
    // OpenCV.js cv.resize(src, dst, dsize)
    // Our API: resize_wasm(src, dst_width, dst_height)
    //
    // Reference: https://docs.opencv.org/4.x/dd/d52/tutorial_js_geometric_transformations.html

    let src = create_test_image_rgb();
    let result = resize_wasm(&src, 20, 15).await.unwrap();

    // Verify dimensions match OpenCV.js behavior
    assert_eq!(result.width(), 20);
    assert_eq!(result.height(), 15);
    assert_eq!(result.channels(), src.channels());
}

// ====================
// 8. CUSTOM TESTS
// ====================

#[wasm_bindgen_test]
async fn test_resize_maintains_structure() {
    // Create image with distinct structure
    let width = 20;
    let height = 20;
    let channels = 1;
    let mut data = vec![0u8; width * height * channels];

    // Create vertical stripes
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            data[idx] = if x < width / 2 { 0 } else { 255 };
        }
    }

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();

    // Resize and verify structure is maintained
    let resized = resize_wasm(&src, 40, 40).await.unwrap();
    let resized_data = resized.get_data();

    // Check left half is dark, right half is bright
    let left_avg = resized_data[..400].iter().map(|&x| x as f64).sum::<f64>() / 400.0;
    let right_avg = resized_data[1200..].iter().map(|&x| x as f64).sum::<f64>() / 400.0;

    assert!(
        left_avg < 100.0,
        "Left half should be dark (avg={})",
        left_avg
    );
    assert!(
        right_avg > 150.0,
        "Right half should be bright (avg={})",
        right_avg
    );
}

#[wasm_bindgen_test]
async fn test_resize_sequence() {
    // Resize multiple times in sequence
    let src = create_test_image_rgb(); // 10x10

    let r1 = resize_wasm(&src, 20, 20).await.unwrap(); // 2x upscale
    assert_eq!(r1.width(), 20);

    let r2 = resize_wasm(&r1, 15, 15).await.unwrap(); // Downscale
    assert_eq!(r2.width(), 15);

    let r3 = resize_wasm(&r2, 30, 10).await.unwrap(); // Non-uniform
    assert_eq!(r3.width(), 30);
    assert_eq!(r3.height(), 10);
}

#[wasm_bindgen_test]
async fn test_resize_uniform_image() {
    // Uniform image should remain uniform after resize
    let width = 10;
    let height = 10;
    let channels = 3;
    let data = vec![128u8; width * height * channels];

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let resized = resize_wasm(&src, 20, 20).await.unwrap();

    let resized_data = resized.get_data();

    // All pixels should be close to 128
    for &pixel in resized_data.iter() {
        let diff = (pixel as i32 - 128).abs();
        assert!(diff < 5, "Uniform image should remain uniform (pixel={})", pixel);
    }
}
