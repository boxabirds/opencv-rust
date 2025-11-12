//! WASM tests for gaussian_blur operation
//!
//! Tests OpenCV.js API parity for the gaussian_blur function

#![cfg(all(target_arch = "wasm32", feature = "wasm"))]

use wasm_bindgen_test::*;
use opencv_rust::wasm::{WasmMat, gaussian_blur_wasm, set_backend_wasm};

mod wasm_test_utils;
use wasm_test_utils::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
async fn test_gaussian_blur_basic_smoke() {
    // Basic smoke test - function doesn't panic
    let src = create_test_image_rgb();
    let result = gaussian_blur_wasm(&src, 5, 1.5).await;
    assert!(result.is_ok(), "gaussian_blur_wasm should not fail on valid input");
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_output_dimensions() {
    // Output should match input dimensions
    let src = create_test_image_rgb();
    let result = gaussian_blur_wasm(&src, 5, 1.5).await.unwrap();

    assert_eq!(result.width(), src.width(), "Width should be preserved");
    assert_eq!(result.height(), src.height(), "Height should be preserved");
    assert_eq!(result.channels(), src.channels(), "Channels should be preserved");
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_actually_blurs() {
    // Blurred image should have lower standard deviation
    let src = create_test_image_rgb();
    let src_stddev = pixel_stddev(&src);

    let blurred = gaussian_blur_wasm(&src, 5, 1.5).await.unwrap();
    let blurred_stddev = pixel_stddev(&blurred);

    assert!(
        blurred_stddev < src_stddev,
        "Blurred image should have lower stddev ({}) than original ({})",
        blurred_stddev,
        src_stddev
    );
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_kernel_size_3() {
    // Test with small kernel size
    let src = create_test_image_rgb();
    let result = gaussian_blur_wasm(&src, 3, 1.0).await;

    assert!(result.is_ok(), "Should handle kernel size 3");
    let output = result.unwrap();
    assert_eq!(output.width(), src.width());
    assert_eq!(output.height(), src.height());
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_kernel_size_7() {
    // Test with larger kernel size
    let src = create_test_image_rgb();
    let result = gaussian_blur_wasm(&src, 7, 2.0).await;

    assert!(result.is_ok(), "Should handle kernel size 7");
    let output = result.unwrap();
    assert_eq!(output.width(), src.width());
    assert_eq!(output.height(), src.height());
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_kernel_size_11() {
    // Test with even larger kernel
    let src = create_test_image_rgb();
    let result = gaussian_blur_wasm(&src, 11, 3.0).await;

    assert!(result.is_ok(), "Should handle kernel size 11");
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_increasing_blur() {
    // Larger kernel = more blur = lower stddev
    let src = create_test_image_rgb();

    let blur3 = gaussian_blur_wasm(&src, 3, 1.0).await.unwrap();
    let blur5 = gaussian_blur_wasm(&src, 5, 1.5).await.unwrap();
    let blur7 = gaussian_blur_wasm(&src, 7, 2.0).await.unwrap();

    let stddev3 = pixel_stddev(&blur3);
    let stddev5 = pixel_stddev(&blur5);
    let stddev7 = pixel_stddev(&blur7);

    assert!(
        stddev7 < stddev5 && stddev5 < stddev3,
        "Larger kernel should produce more blur: {} < {} < {}",
        stddev7,
        stddev5,
        stddev3
    );
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_sigma_zero() {
    // Sigma = 0 should use automatic calculation
    let src = create_test_image_rgb();
    let result = gaussian_blur_wasm(&src, 5, 0.0).await;

    assert!(result.is_ok(), "Should handle sigma=0 (auto calculation)");
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_small_sigma() {
    // Small sigma = less blur
    let src = create_test_image_rgb();
    let result = gaussian_blur_wasm(&src, 5, 0.5).await;

    assert!(result.is_ok(), "Should handle small sigma");
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_large_sigma() {
    // Large sigma = more blur
    let src = create_test_image_rgb();
    let result = gaussian_blur_wasm(&src, 9, 5.0).await;

    assert!(result.is_ok(), "Should handle large sigma");
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_grayscale() {
    // Test with grayscale image
    let src = create_test_image_gray();
    let result = gaussian_blur_wasm(&src, 5, 1.5).await.unwrap();

    assert_eq!(result.width(), src.width());
    assert_eq!(result.height(), src.height());
    assert_eq!(result.channels(), 1, "Should preserve grayscale");
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_preserves_average() {
    // Gaussian blur should preserve average intensity
    let src = create_test_image_rgb();
    let src_avg = average_pixel_value(&src);

    let blurred = gaussian_blur_wasm(&src, 5, 1.5).await.unwrap();
    let blurred_avg = average_pixel_value(&blurred);

    let diff = (src_avg - blurred_avg).abs();
    assert!(
        diff < 5.0,
        "Average pixel value should be preserved (diff={})",
        diff
    );
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_smoothness() {
    // Blurred image should be smoother (less high-frequency content)
    // We can approximate this by checking neighboring pixel differences
    let src = create_test_image_rgb();
    let blurred = gaussian_blur_wasm(&src, 5, 1.5).await.unwrap();

    let src_data = src.get_data();
    let blur_data = blurred.get_data();

    // Calculate average neighbor difference for first row
    let width = src.width();
    let channels = src.channels();

    let mut src_diff_sum = 0i32;
    let mut blur_diff_sum = 0i32;

    for i in 0..(width - 1) * channels {
        src_diff_sum += (src_data[i] as i32 - src_data[i + channels] as i32).abs();
        blur_diff_sum += (blur_data[i] as i32 - blur_data[i + channels] as i32).abs();
    }

    assert!(
        blur_diff_sum < src_diff_sum,
        "Blurred image should have smoother transitions"
    );
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_cpu_backend() {
    // Test with explicit CPU backend
    set_backend_wasm("cpu");

    let src = create_test_image_rgb();
    let result = gaussian_blur_wasm(&src, 5, 1.5).await;

    assert!(result.is_ok(), "gaussian_blur_wasm should work with CPU backend");

    // Reset to auto
    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_gpu_backend() {
    // Test with explicit WebGPU backend
    set_backend_wasm("webgpu");

    let src = create_test_image_rgb();
    let result = gaussian_blur_wasm(&src, 5, 1.5).await;

    // If GPU backend is set, operation MUST succeed or provide clear error
    match result {
        Ok(output) => {
            // GPU is available and working - verify output
            assert_eq!(output.width(), src.width(), "GPU: Width should be preserved");
            assert_eq!(output.height(), src.height(), "GPU: Height should be preserved");
            assert_eq!(output.channels(), src.channels(), "GPU: Channels should be preserved");
            web_sys::console::log_1(&"✓ GPU backend test passed".into());
        }
        Err(e) => {
            // GPU failed - this is acceptable ONLY if GPU is not available
            let error_msg = format!("{:?}", e);
            if error_msg.contains("GPU not available") || error_msg.contains("WebGPU not supported") {
                web_sys::console::log_1(&"⚠ GPU not available, skipping GPU test".into());
            } else {
                // GPU is available but broken - FAIL THE TEST
                panic!("GPU backend failed with error (this should not happen if GPU initialized): {:?}", e);
            }
        }
    }

    // Reset to auto
    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_cpu_gpu_similarity() {
    // CPU and GPU should produce similar results (within tolerance for floating point)
    let src = create_test_image_rgb();

    // Test with CPU
    set_backend_wasm("cpu");
    let cpu_result = gaussian_blur_wasm(&src, 5, 1.5).await
        .expect("CPU backend must work");

    // Test with GPU - only compare if GPU is available
    set_backend_wasm("webgpu");
    let gpu_result = gaussian_blur_wasm(&src, 5, 1.5).await;

    if let Ok(gpu_output) = gpu_result {
        // GPU is available - verify CPU and GPU produce similar results
        assert!(
            images_are_similar(&cpu_result, &gpu_output, 5.0),
            "CPU and GPU blur results should be similar (tolerance=5.0)"
        );
        web_sys::console::log_1(&"✓ CPU/GPU similarity test passed".into());
    } else {
        web_sys::console::log_1(&"⚠ GPU not available, skipping CPU/GPU comparison".into());
    }

    // Reset to auto
    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_large_image() {
    // Test with larger image
    let src = create_test_image_large();
    let result = gaussian_blur_wasm(&src, 5, 1.5).await;

    assert!(result.is_ok(), "Should handle large images");

    let output = result.unwrap();
    assert_eq!(output.width(), src.width());
    assert_eq!(output.height(), src.height());
    assert_eq!(output.channels(), src.channels());
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_uniform_image() {
    // Blurring a uniform image should return same uniform image
    let width = 10;
    let height = 10;
    let channels = 3;
    let data = vec![128u8; width * height * channels];

    let src = WasmMat::from_image_data(&data, width, height, channels).unwrap();
    let blurred = gaussian_blur_wasm(&src, 5, 1.5).await.unwrap();

    let blurred_data = blurred.get_data();

    // All pixels should remain close to 128
    for &pixel in blurred_data.iter() {
        let diff = (pixel as i32 - 128).abs();
        assert!(diff < 2, "Uniform image should remain uniform after blur");
    }
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_repeated() {
    // Applying blur twice should give more blur than once
    let src = create_test_image_rgb();

    let blur_once = gaussian_blur_wasm(&src, 5, 1.5).await.unwrap();
    let blur_twice = gaussian_blur_wasm(&blur_once, 5, 1.5).await.unwrap();

    let stddev_once = pixel_stddev(&blur_once);
    let stddev_twice = pixel_stddev(&blur_twice);

    assert!(
        stddev_twice < stddev_once,
        "Applying blur twice should reduce stddev more: {} < {}",
        stddev_twice,
        stddev_once
    );
}

#[wasm_bindgen_test]
async fn test_gaussian_blur_idempotency_limit() {
    // After many blur operations, image should converge to uniform
    let src = create_test_image_rgb();
    let mut current = gaussian_blur_wasm(&src, 7, 2.0).await.unwrap();

    // Apply blur 5 times
    for _ in 0..5 {
        current = gaussian_blur_wasm(&current, 7, 2.0).await.unwrap();
    }

    // After many blurs, stddev should be very low (nearly uniform)
    let final_stddev = pixel_stddev(&current);
    assert!(
        final_stddev < 30.0,
        "After many blurs, image should be nearly uniform (stddev={})",
        final_stddev
    );
}
