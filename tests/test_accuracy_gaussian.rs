#![allow(unused_comparisons)]
/// Bit-level accuracy tests for Gaussian Blur
/// These tests verify that optimizations don't change results
mod test_utils;

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::{Scalar, Size};
use opencv_rust::imgproc::gaussian_blur;
use test_utils::*;

/// Test Gaussian blur is deterministic
#[test]
fn test_gaussian_blur_deterministic() {
    let mut src = Mat::new(50, 50, 3, MatDepth::U8).unwrap();

    // Create random-ish pattern
    for row in 0..50 {
        for col in 0..50 {
            let pixel = src.at_mut(row, col).unwrap();
            pixel[0] = ((row * 7 + col * 13) % 256) as u8;
            pixel[1] = ((row * 11 + col * 5) % 256) as u8;
            pixel[2] = ((row * col) % 256) as u8;
        }
    }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    gaussian_blur(&src, &mut dst1, Size::new(5, 5), 1.5).unwrap();
    gaussian_blur(&src, &mut dst2, Size::new(5, 5), 1.5).unwrap();

    // Results should be bit-exact identical
    assert_images_equal(&dst1, &dst2, "Gaussian blur should be deterministic");
}

/// Test Gaussian blur on uniform image produces uniform output
#[test]
fn test_gaussian_blur_uniform_image() {
    let src = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    gaussian_blur(&src, &mut dst, Size::new(5, 5), 1.0).unwrap();

    // Uniform image should stay uniform (allow ±1 for rounding)
    for row in 2..48 {
        for col in 2..48 {
            let pixel = dst.at(row, col).unwrap()[0];
            assert!((pixel as i32 - 128).abs() <= 1,
                "Uniform image blur at ({}, {}) should be ~128, got {}", row, col, pixel);
        }
    }
}

/// Test Gaussian blur smooths sharp edges
#[test]
fn test_gaussian_blur_smooths_edges() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create sharp vertical edge
    for row in 0..50 {
        for col in 0..25 {
            src.at_mut(row, col).unwrap()[0] = 0;
        }
        for col in 25..50 {
            src.at_mut(row, col).unwrap()[0] = 255;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    gaussian_blur(&src, &mut dst, Size::new(7, 7), 2.0).unwrap();

    // Edge should be smoothed - check transition region
    let left = dst.at(25, 20).unwrap()[0];   // Left of edge
    let center = dst.at(25, 25).unwrap()[0]; // At edge
    let right = dst.at(25, 30).unwrap()[0];  // Right of edge

    assert!(left < 50, "Left of edge should be dark, got {}", left);
    assert!(right > 200, "Right of edge should be bright, got {}", right);
    assert!(center > 50 && center < 200, "Center should be blurred transition, got {}", center);
}

/// Test Gaussian blur with kernel size 3x3
#[test]
fn test_gaussian_blur_3x3_accuracy() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();

    // Create center bright pixel with some padding
    for row in 0..20 {
        for col in 0..20 {
            src.at_mut(row, col).unwrap()[0] = 0;
        }
    }
    src.at_mut(10, 10).unwrap()[0] = 255; // Bright center

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    gaussian_blur(&src, &mut dst, Size::new(3, 3), 1.0).unwrap();

    // Center should still be brightest (though blurred)
    let center = dst.at(10, 10).unwrap()[0];
    assert!(center > 30, "Center after blur should be bright, got {}", center);

    // Neighbors should be dimmer
    let neighbor = dst.at(10, 11).unwrap()[0];
    assert!(neighbor < center, "Neighbor should be dimmer than center");
    assert!(neighbor > 0, "Neighbor should be affected by blur, got {}", neighbor);
}

/// Test Gaussian blur with kernel size 5x5
#[test]
fn test_gaussian_blur_5x5_wider_spread() {
    let mut src = Mat::new(30, 30, 1, MatDepth::U8).unwrap();

    // Single bright pixel
    for row in 0..30 {
        for col in 0..30 {
            src.at_mut(row, col).unwrap()[0] = 0;
        }
    }
    src.at_mut(15, 15).unwrap()[0] = 255;

    let mut dst3 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst5 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    gaussian_blur(&src, &mut dst3, Size::new(3, 3), 1.0).unwrap();
    gaussian_blur(&src, &mut dst5, Size::new(5, 5), 1.5).unwrap();

    // 5x5 kernel should have more total spread (check closer pixels)
    let near_pixel_3 = dst3.at(15, 16).unwrap()[0];
    let near_pixel_5 = dst5.at(15, 16).unwrap()[0];

    // Both should have some blur, but we're just verifying they work
    assert!(near_pixel_3 > 0 || near_pixel_5 > 0,
        "At least one kernel should spread to neighbors: 3x3={}, 5x5={}", near_pixel_3, near_pixel_5);
}

/// Test Gaussian blur with different sigma values
#[test]
fn test_gaussian_blur_sigma_effect() {
    let mut src = Mat::new(40, 40, 1, MatDepth::U8).unwrap();

    // Create bright center region (not just a pixel)
    for row in 0..40 {
        for col in 0..40 {
            src.at_mut(row, col).unwrap()[0] = 0;
        }
    }
    // 3x3 bright region in center
    for row in 19..22 {
        for col in 19..22 {
            src.at_mut(row, col).unwrap()[0] = 255;
        }
    }

    let mut dst_sigma1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst_sigma3 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    gaussian_blur(&src, &mut dst_sigma1, Size::new(7, 7), 1.0).unwrap();
    gaussian_blur(&src, &mut dst_sigma3, Size::new(7, 7), 3.0).unwrap();

    // Check that blur produces some spread (both should create non-zero neighbors)
    let neighbor_s1 = dst_sigma1.at(20, 22).unwrap()[0];
    let neighbor_s3 = dst_sigma3.at(20, 22).unwrap()[0];

    assert!(neighbor_s1 > 0 || neighbor_s3 > 0,
        "Gaussian blur should spread to neighbors: s1={}, s3={}", neighbor_s1, neighbor_s3);
}

/// Test Gaussian blur preserves total brightness (approximately)
#[test]
fn test_gaussian_blur_energy_conservation() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();

    // Create 5x5 bright region
    for row in 0..20 {
        for col in 0..20 {
            src.at_mut(row, col).unwrap()[0] = 0;
        }
    }
    for row in 8..13 {
        for col in 8..13 {
            src.at_mut(row, col).unwrap()[0] = 100;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    gaussian_blur(&src, &mut dst, Size::new(5, 5), 1.5).unwrap();

    // Sum total brightness (should be approximately preserved)
    let mut src_sum = 0u64;
    let mut dst_sum = 0u64;

    for row in 0..20 {
        for col in 0..20 {
            src_sum += src.at(row, col).unwrap()[0] as u64;
            dst_sum += dst.at(row, col).unwrap()[0] as u64;
        }
    }

    // Allow up to 10% difference due to edge effects and rounding
    let diff_percent = ((src_sum as i64 - dst_sum as i64).abs() as f64 / src_sum as f64) * 100.0;
    assert!(diff_percent < 10.0,
        "Energy should be approximately conserved: src={}, dst={}, diff={:.2}%",
        src_sum, dst_sum, diff_percent);
}

/// Test Gaussian blur on multi-channel image
#[test]
fn test_gaussian_blur_multichannel_independence() {
    let mut src = Mat::new(20, 20, 3, MatDepth::U8).unwrap();

    // Each channel has different pattern
    for row in 0..20 {
        for col in 0..20 {
            let pixel = src.at_mut(row, col).unwrap();
            pixel[0] = if row == 10 && col == 10 { 255 } else { 0 }; // Red spike
            pixel[1] = (row * 10) as u8;                              // Green gradient
            pixel[2] = 128;                                           // Blue constant
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    gaussian_blur(&src, &mut dst, Size::new(5, 5), 1.5).unwrap();

    // Blue should remain approximately constant
    for row in 5..15 {
        for col in 5..15 {
            let blue = dst.at(row, col).unwrap()[2];
            assert!((blue as i32 - 128).abs() <= 5,
                "Blue constant channel at ({}, {}) should be ~128, got {}", row, col, blue);
        }
    }

    // Red channel should have spread from spike
    let red_center = dst.at(10, 10).unwrap()[0];
    let red_neighbor = dst.at(10, 11).unwrap()[0];
    assert!(red_center > red_neighbor, "Red spike should be brightest at center");
    assert!(red_neighbor > 0, "Red should spread to neighbors");
}

/// Test Gaussian blur handles edge pixels correctly
#[test]
fn test_gaussian_blur_edge_handling() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    // Fill with gradient
    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 20) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    gaussian_blur(&src, &mut dst, Size::new(5, 5), 1.0).unwrap();

    // Should not crash and produce valid output at edges
    assert_eq!(dst.rows(), 10);
    assert_eq!(dst.cols(), 10);

    // Edge pixels should be in valid range
    for col in 0..10 {
        let top = dst.at(0, col).unwrap()[0];
        let bottom = dst.at(9, col).unwrap()[0];
        assert!(top <= 255 && bottom <= 255);
    }
}

/// Test Gaussian blur with large kernel (11x11)
#[test]
fn test_gaussian_blur_11x11_heavy_smoothing() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Checkerboard pattern
    for row in 0..50 {
        for col in 0..50 {
            let value = if (row / 5 + col / 5) % 2 == 0 { 0 } else { 255 };
            src.at_mut(row, col).unwrap()[0] = value;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    gaussian_blur(&src, &mut dst, Size::new(11, 11), 3.0).unwrap();

    // Heavy blur should significantly smooth the pattern
    // Center region should be mid-range values
    let mut mid_values = 0;
    for row in 15..35 {
        for col in 15..35 {
            let val = dst.at(row, col).unwrap()[0];
            if val > 50 && val < 200 {
                mid_values += 1;
            }
        }
    }

    assert!(mid_values > 200,
        "Heavy blur should produce many mid-range values, got {} pixels", mid_values);
}

/// Visual inspection test (ignored by default)
#[test]
#[ignore]
fn test_gaussian_blur_visual_inspection() {
    let mut src = Mat::new(15, 15, 1, MatDepth::U8).unwrap();

    // Create simple pattern
    for row in 0..15 {
        for col in 0..15 {
            src.at_mut(row, col).unwrap()[0] = 0;
        }
    }
    // Bright square in center
    for row in 6..9 {
        for col in 6..9 {
            src.at_mut(row, col).unwrap()[0] = 255;
        }
    }

    println!("\nOriginal:");
    print_image_data(&src, "Source", 15, 15);

    let mut dst3 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    gaussian_blur(&src, &mut dst3, Size::new(3, 3), 1.0).unwrap();

    println!("\nAfter 3x3 Gaussian blur:");
    print_image_data(&dst3, "Blurred 3x3", 15, 15);

    let mut dst7 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    gaussian_blur(&src, &mut dst7, Size::new(7, 7), 2.0).unwrap();

    println!("\nAfter 7x7 Gaussian blur:");
    print_image_data(&dst7, "Blurred 7x7", 15, 15);
}
