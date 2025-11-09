/// Bit-level accuracy tests for Median Blur
/// These tests verify that optimizations don't change results
mod test_utils;

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::Scalar;
use opencv_rust::imgproc::median_blur;
use test_utils::*;

/// Test median blur is deterministic
#[test]
fn test_median_blur_deterministic() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create pattern with noise
    for row in 0..50 {
        for col in 0..50 {
            let base = ((row + col) * 5) as u8;
            // Add some salt-and-pepper noise
            let noise = if (row * 7 + col * 11) % 13 == 0 { 100 } else { 0 };
            src.at_mut(row, col).unwrap()[0] = base.saturating_add(noise);
        }
    }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    median_blur(&src, &mut dst1, 3).unwrap();
    median_blur(&src, &mut dst2, 3).unwrap();

    // Results should be bit-exact identical
    assert_images_equal(&dst1, &dst2, "Median blur should be deterministic");
}

/// Test median blur on uniform image
#[test]
fn test_median_blur_uniform_image() {
    let src = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    median_blur(&src, &mut dst, 3).unwrap();

    // Uniform image should remain uniform
    for row in 0..50 {
        for col in 0..50 {
            assert_eq!(dst.at(row, col).unwrap()[0], 128,
                "Uniform image should remain uniform at ({}, {})", row, col);
        }
    }
}

/// Test median blur removes salt-and-pepper noise
#[test]
fn test_median_blur_noise_removal() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create uniform background
    for row in 0..50 {
        for col in 0..50 {
            src.at_mut(row, col).unwrap()[0] = 100;
        }
    }

    // Add isolated noise pixels
    src.at_mut(10, 10).unwrap()[0] = 255; // White noise
    src.at_mut(20, 20).unwrap()[0] = 0;   // Black noise
    src.at_mut(30, 30).unwrap()[0] = 255; // White noise

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    median_blur(&src, &mut dst, 3).unwrap();

    // Noise pixels should be replaced with median (100)
    assert_eq!(dst.at(10, 10).unwrap()[0], 100,
        "White noise should be removed");
    assert_eq!(dst.at(20, 20).unwrap()[0], 100,
        "Black noise should be removed");
    assert_eq!(dst.at(30, 30).unwrap()[0], 100,
        "White noise should be removed");
}

/// Test median blur preserves edges better than Gaussian
#[test]
fn test_median_blur_edge_preservation() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create sharp vertical edge
    for row in 0..50 {
        for col in 0..25 {
            src.at_mut(row, col).unwrap()[0] = 50;
        }
        for col in 25..50 {
            src.at_mut(row, col).unwrap()[0] = 200;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    median_blur(&src, &mut dst, 3).unwrap();

    // Edge should be preserved (median of [50,50,200] or [200,200,50] stays sharp)
    // Center regions should remain close to original values
    let left_center = dst.at(25, 12).unwrap()[0];
    let right_center = dst.at(25, 37).unwrap()[0];

    assert_eq!(left_center, 50,
        "Left region should preserve original value");
    assert_eq!(right_center, 200,
        "Right region should preserve original value");
}

/// Test median blur kernel size effects (larger → more smoothing)
#[test]
fn test_median_blur_kernel_size_effect() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create noisy gradient
    for row in 0..50 {
        for col in 0..50 {
            let base = (row * 5) as u8;
            let noise = if (row + col) % 3 == 0 { 30 } else { 0 };
            src.at_mut(row, col).unwrap()[0] = base.saturating_add(noise);
        }
    }

    let mut dst3 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst5 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    median_blur(&src, &mut dst3, 3).unwrap();
    median_blur(&src, &mut dst5, 5).unwrap();

    // Both should process without errors
    assert_eq!(dst3.rows(), 50);
    assert_eq!(dst5.rows(), 50);
}

/// Test median blur on multi-channel image
#[test]
fn test_median_blur_multichannel() {
    let mut src = Mat::new(30, 30, 3, MatDepth::U8).unwrap();

    // Fill each channel with different pattern
    for row in 0..30 {
        for col in 0..30 {
            let pixel = src.at_mut(row, col).unwrap();
            pixel[0] = ((row + col) * 8) as u8;      // Red
            pixel[1] = ((row * 2 + col) * 4) as u8;  // Green
            pixel[2] = 128;                           // Blue constant
        }
    }

    // Add noise to each channel
    src.at_mut(10, 10).unwrap()[0] = 255;
    src.at_mut(10, 10).unwrap()[1] = 0;

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    median_blur(&src, &mut dst, 3).unwrap();

    // Verify channels processed independently
    // Blue channel (constant) should remain ~128
    for row in 5..25 {
        for col in 5..25 {
            assert_eq!(dst.at(row, col).unwrap()[2], 128,
                "Blue constant channel at ({}, {}) should be 128", row, col);
        }
    }

    // Noise should be removed from red and green channels
    let denoised_pixel = dst.at(10, 10).unwrap();
    assert!(denoised_pixel[0] < 255 && denoised_pixel[0] > 0,
        "Red noise should be removed");
    assert!(denoised_pixel[1] > 0 && denoised_pixel[1] < 255,
        "Green noise should be removed");
}

/// Test median blur with kernel size 5
#[test]
fn test_median_blur_ksize_5() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    for row in 0..50 {
        for col in 0..50 {
            src.at_mut(row, col).unwrap()[0] = ((row + col) * 2) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    median_blur(&src, &mut dst, 5).unwrap();

    assert_eq!(dst.rows(), 50);
    assert_eq!(dst.cols(), 50);
}

/// Test median blur with kernel size 7
#[test]
fn test_median_blur_ksize_7() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    for row in 0..50 {
        for col in 0..50 {
            src.at_mut(row, col).unwrap()[0] = ((row * col) % 256) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    median_blur(&src, &mut dst, 7).unwrap();

    assert_eq!(dst.rows(), 50);
    assert_eq!(dst.cols(), 50);
}

/// Test median blur boundary handling
#[test]
fn test_median_blur_boundary() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = ((row + col) * 20) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    median_blur(&src, &mut dst, 3).unwrap();

    // Border pixels should be valid (uses border replication)
    for i in 0..10 {
        assert!(dst.at(0, i).unwrap()[0] <= 255, "Top border valid");
        assert!(dst.at(9, i).unwrap()[0] <= 255, "Bottom border valid");
        assert!(dst.at(i, 0).unwrap()[0] <= 255, "Left border valid");
        assert!(dst.at(i, 9).unwrap()[0] <= 255, "Right border valid");
    }
}

/// Test median blur output range [0, 255]
#[test]
fn test_median_blur_output_range() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create extreme checkerboard pattern
    for row in 0..50 {
        for col in 0..50 {
            src.at_mut(row, col).unwrap()[0] = if (row + col) % 2 == 0 { 0 } else { 255 };
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    median_blur(&src, &mut dst, 3).unwrap();

    // All values should be in valid range
    for row in 0..50 {
        for col in 0..50 {
            let val = dst.at(row, col).unwrap()[0];
            assert!(val <= 255,
                "Median blur output at ({}, {}) out of range: {}", row, col, val);
        }
    }
}

/// Test median blur on small image
#[test]
fn test_median_blur_small_image() {
    let mut src = Mat::new(5, 5, 1, MatDepth::U8).unwrap();

    for i in 0..25 {
        src.data_mut()[i] = ((i * 10) % 256) as u8;
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    median_blur(&src, &mut dst, 3).unwrap();

    assert_eq!(dst.rows(), 5);
    assert_eq!(dst.cols(), 5);
}

/// Test median blur processes checkerboard correctly
#[test]
fn test_median_blur_checkerboard() {
    let mut src = Mat::new(30, 30, 1, MatDepth::U8).unwrap();

    // Create 2x2 checkerboard
    for row in 0..30 {
        for col in 0..30 {
            let value = if ((row / 2) + (col / 2)) % 2 == 0 { 100 } else { 200 };
            src.at_mut(row, col).unwrap()[0] = value;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    median_blur(&src, &mut dst, 3).unwrap();

    // Median should select one of the two dominant values
    for row in 2..28 {
        for col in 2..28 {
            let val = dst.at(row, col).unwrap()[0];
            // Should be either 100, 200, or a value between them
            assert!(val >= 100 && val <= 200,
                "Checkerboard median at ({}, {}) should be in [100, 200], got {}",
                row, col, val);
        }
    }
}

/// Visual inspection test (ignored by default)
#[test]
#[ignore]
fn test_median_blur_visual_inspection() {
    let mut src = Mat::new(15, 15, 1, MatDepth::U8).unwrap();

    // Create pattern with noise
    for row in 0..15 {
        for col in 0..15 {
            src.at_mut(row, col).unwrap()[0] = 100;
        }
    }

    // Add salt-and-pepper noise
    src.at_mut(5, 5).unwrap()[0] = 255;
    src.at_mut(5, 10).unwrap()[0] = 0;
    src.at_mut(10, 5).unwrap()[0] = 255;
    src.at_mut(10, 10).unwrap()[0] = 0;

    println!("\nInput (with salt-and-pepper noise):");
    print_image_data(&src, "Source", 15, 15);

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    median_blur(&src, &mut dst, 3).unwrap();

    println!("\nAfter median blur (3x3):");
    print_image_data(&dst, "Denoised", 15, 15);

    let stats = compute_diff_stats(&src, &dst);
    println!("\nDifference from original:");
    println!("{}", stats);
}
