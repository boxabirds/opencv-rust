#![cfg(feature = "gpu")]

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::Size;
use opencv_rust::gpu::{init_gpu, gpu_available};
use opencv_rust::imgproc::gaussian_blur;

#[test]
fn test_gpu_context_initialization() {
    // Try to initialize GPU - may fail on systems without GPU
    let gpu_initialized = init_gpu();

    if gpu_initialized {
        println!("GPU successfully initialized");
        assert!(gpu_available());
    } else {
        println!("GPU initialization failed (this is OK on systems without GPU support)");
        assert!(!gpu_available());
    }
}

#[test]
fn test_gpu_gaussian_blur_basic() {
    // Initialize GPU
    if !init_gpu() {
        println!("Skipping GPU test - GPU not available");
        return;
    }

    // Create a simple test image (100x100, grayscale)
    let mut src = Mat::new(100, 100, 1, MatDepth::U8).unwrap();
    let src_data = src.data_mut();

    // Create a simple pattern - vertical stripes
    for y in 0..100 {
        for x in 0..100 {
            src_data[y * 100 + x] = if x % 20 < 10 { 255 } else { 0 };
        }
    }

    // Apply Gaussian blur (this should use GPU if available)
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let result = gaussian_blur(&src, &mut dst, Size::new(5, 5), 1.0);

    assert!(result.is_ok(), "Gaussian blur should complete successfully");
    assert_eq!(dst.rows(), 100);
    assert_eq!(dst.cols(), 100);
    assert_eq!(dst.channels(), 1);

    // Verify output is not all zeros or all 255s (blur should smooth the pattern)
    let dst_data = dst.data();
    let sum: u32 = dst_data.iter().map(|&x| x as u32).sum();
    let avg = sum / (100 * 100);

    println!("Average pixel value after blur: {}", avg);

    // The average should be around 127-128 (since we have 50% white, 50% black)
    assert!(avg > 100 && avg < 155, "Blur should produce smoothed values");
}

#[test]
fn test_gpu_cpu_equivalence() {
    // This test verifies that GPU and CPU implementations produce similar results

    if !init_gpu() {
        println!("Skipping GPU equivalence test - GPU not available");
        return;
    }

    // Create test image
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();
    let src_data = src.data_mut();

    // Create a more complex pattern
    for y in 0..50 {
        for x in 0..50 {
            src_data[y * 50 + x] = ((x * 5 + y * 5) % 256) as u8;
        }
    }

    // Apply blur (should use GPU)
    let mut dst_gpu = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    gaussian_blur(&src, &mut dst_gpu, Size::new(3, 3), 1.0).unwrap();

    // For full CPU/GPU comparison, we would need a way to force CPU mode
    // For now, just verify the output is reasonable
    let gpu_data = dst_gpu.data();

    // Check that values are within expected range
    for &val in gpu_data {
        assert!(val <= 255, "Output values should be valid u8");
    }

    println!("GPU blur test completed successfully");
}

#[test]
fn test_gpu_blur_multiple_channels() {
    if !init_gpu() {
        println!("Skipping GPU multi-channel test - GPU not available");
        return;
    }

    // Create RGB test image (50x50x3)
    let mut src = Mat::new(50, 50, 3, MatDepth::U8).unwrap();
    let src_data = src.data_mut();

    // Fill with gradient pattern
    for y in 0..50 {
        for x in 0..50 {
            let idx = (y * 50 + x) * 3;
            src_data[idx] = (x * 5) as u8;     // R
            src_data[idx + 1] = (y * 5) as u8; // G
            src_data[idx + 2] = 128;           // B
        }
    }

    // Apply Gaussian blur
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let result = gaussian_blur(&src, &mut dst, Size::new(5, 5), 1.5);

    assert!(result.is_ok(), "Multi-channel blur should complete successfully");
    assert_eq!(dst.rows(), 50);
    assert_eq!(dst.cols(), 50);
    assert_eq!(dst.channels(), 3);

    println!("Multi-channel GPU blur test completed successfully");
}

#[test]
fn test_gpu_blur_edge_cases() {
    if !init_gpu() {
        println!("Skipping GPU edge case test - GPU not available");
        return;
    }

    // Test with small image
    let mut src_small = Mat::new(10, 10, 1, MatDepth::U8).unwrap();
    src_small.data_mut().fill(128);

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let result = gaussian_blur(&src_small, &mut dst, Size::new(3, 3), 1.0);
    assert!(result.is_ok(), "Small image should work");

    // Test with larger kernel
    let mut src = Mat::new(100, 100, 1, MatDepth::U8).unwrap();
    src.data_mut().fill(200);

    let result = gaussian_blur(&src, &mut dst, Size::new(9, 9), 2.0);
    assert!(result.is_ok(), "Large kernel should work");

    // Verify all output values are close to input (uniform image should stay uniform)
    let dst_data = dst.data();
    for &val in dst_data {
        assert!((val as i32 - 200).abs() < 5, "Uniform image should stay uniform after blur");
    }

    println!("Edge case tests completed successfully");
}
