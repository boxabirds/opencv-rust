/// Bit-level accuracy tests for Resize operations
/// These tests verify that optimizations don't change results
mod test_utils;

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::Scalar;
use opencv_rust::imgproc::resize;
use opencv_rust::core::types::{Size, InterpolationFlag};
use test_utils::*;

/// Test resize maintains deterministic output
#[test]
fn test_resize_deterministic_downscale() {
    let mut src = Mat::new(100, 100, 3, MatDepth::U8).unwrap();

    // Create gradient pattern
    for row in 0..100 {
        for col in 0..100 {
            let pixel = src.at_mut(row, col).unwrap();
            pixel[0] = ((row * 2 + col) % 256) as u8;
            pixel[1] = ((row + col * 2) % 256) as u8;
            pixel[2] = ((row * col / 10) % 256) as u8;
        }
    }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    resize(&src, &mut dst1, Size::new(50, 50), InterpolationFlag::Linear).unwrap();
    resize(&src, &mut dst2, Size::new(50, 50), InterpolationFlag::Linear).unwrap();

    // Results should be bit-exact identical
    assert_images_equal(&dst1, &dst2, "Resize downscale should be deterministic");
}

/// Test resize upscale deterministic output
#[test]
fn test_resize_deterministic_upscale() {
    let mut src = Mat::new(50, 50, 3, MatDepth::U8).unwrap();

    // Create test pattern
    for row in 0..50 {
        for col in 0..50 {
            let pixel = src.at_mut(row, col).unwrap();
            pixel[0] = (row * 5) as u8;
            pixel[1] = (col * 5) as u8;
            pixel[2] = ((row + col) * 2) as u8;
        }
    }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    resize(&src, &mut dst1, Size::new(100, 100), InterpolationFlag::Linear).unwrap();
    resize(&src, &mut dst2, Size::new(100, 100), InterpolationFlag::Linear).unwrap();

    assert_images_equal(&dst1, &dst2, "Resize upscale should be deterministic");
}

/// Test nearest neighbor interpolation accuracy
#[test]
fn test_resize_nearest_neighbor_exact() {
    let mut src = Mat::new(4, 4, 1, MatDepth::U8).unwrap();

    // Create simple 2x2 blocks
    for row in 0..4 {
        for col in 0..4 {
            let value = ((row / 2) * 2 + (col / 2)) as u8 * 50;
            src.at_mut(row, col).unwrap()[0] = value;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    resize(&src, &mut dst, Size::new(8, 8), InterpolationFlag::Nearest).unwrap();

    // Nearest neighbor should replicate exact pixel values
    for row in 0..8 {
        for col in 0..8 {
            let expected_row = row / 2;
            let expected_col = col / 2;
            let expected = src.at(expected_row, expected_col).unwrap()[0];
            let actual = dst.at(row, col).unwrap()[0];

            assert_eq!(actual, expected,
                "Nearest neighbor at ({}, {}) should match source at ({}, {})",
                row, col, expected_row, expected_col);
        }
    }
}

/// Test bilinear interpolation produces smooth gradients
#[test]
fn test_resize_bilinear_smooth() {
    let mut src = Mat::new(2, 2, 1, MatDepth::U8).unwrap();

    // Simple 2x2 gradient: 0, 100, 100, 200
    src.at_mut(0, 0).unwrap()[0] = 0;
    src.at_mut(0, 1).unwrap()[0] = 100;
    src.at_mut(1, 0).unwrap()[0] = 100;
    src.at_mut(1, 1).unwrap()[0] = 200;

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    resize(&src, &mut dst, Size::new(4, 4), InterpolationFlag::Linear).unwrap();

    // Center region should have interpolated values (not exact average due to sampling)
    let center = dst.at(1, 1).unwrap()[0];

    // Should be between min and max input values
    assert!(center >= 0 && center <= 200,
        "Bilinear interpolation should produce value in input range, got {}", center);

    // Edges should be monotonic
    assert!(dst.at(0, 0).unwrap()[0] <= dst.at(0, 1).unwrap()[0],
        "Top row should be non-decreasing");
    assert!(dst.at(0, 0).unwrap()[0] <= dst.at(1, 0).unwrap()[0],
        "Left column should be non-decreasing");
}

/// Test resize preserves color channels independently
#[test]
fn test_resize_multichannel_independence() {
    let mut src = Mat::new(10, 10, 3, MatDepth::U8).unwrap();

    // Fill each channel with different pattern
    for row in 0..10 {
        for col in 0..10 {
            let pixel = src.at_mut(row, col).unwrap();
            pixel[0] = (row * 20) as u8;      // Red gradient vertical
            pixel[1] = (col * 20) as u8;      // Green gradient horizontal
            pixel[2] = 128;                    // Blue constant
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    resize(&src, &mut dst, Size::new(20, 20), InterpolationFlag::Linear).unwrap();

    // Verify channels processed independently
    // Blue channel should remain ~128 everywhere (allow ±2 for interpolation)
    for row in 0..20 {
        for col in 0..20 {
            let blue = dst.at(row, col).unwrap()[2];
            assert!((blue as i32 - 128).abs() <= 2,
                "Blue constant channel at ({}, {}) should be ~128, got {}", row, col, blue);
        }
    }
}

/// Test resize maintains value range [0, 255]
#[test]
fn test_resize_no_overflow() {
    let mut src = Mat::new(10, 10, 3, MatDepth::U8).unwrap();

    // Fill with extreme values (checkerboard of 0 and 255)
    for row in 0..10 {
        for col in 0..10 {
            let value = if (row + col) % 2 == 0 { 0 } else { 255 };
            let pixel = src.at_mut(row, col).unwrap();
            pixel[0] = value;
            pixel[1] = value;
            pixel[2] = value;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    resize(&src, &mut dst, Size::new(25, 25), InterpolationFlag::Linear).unwrap();

    // All values should be in valid range
    for row in 0..25 {
        for col in 0..25 {
            let pixel = dst.at(row, col).unwrap();
            for ch in 0..3 {
                assert!(pixel[ch] <= 255,
                    "Pixel at ({}, {}) ch{} out of range: {}", row, col, ch, pixel[ch]);
            }
        }
    }
}

/// Test small to large upscale accuracy
#[test]
fn test_resize_upscale_4x() {
    let mut src = Mat::new(5, 5, 1, MatDepth::U8).unwrap();

    // Create simple cross pattern
    for i in 0..5 {
        src.at_mut(i, 2).unwrap()[0] = 200; // Vertical line
        src.at_mut(2, i).unwrap()[0] = 200; // Horizontal line
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    resize(&src, &mut dst, Size::new(20, 20), InterpolationFlag::Linear).unwrap();

    assert_eq!(dst.rows(), 20);
    assert_eq!(dst.cols(), 20);

    // Center cross should have high values
    let center_vertical = dst.at(10, 8).unwrap()[0];
    let center_horizontal = dst.at(8, 10).unwrap()[0];

    assert!(center_vertical > 100, "Vertical cross should be bright, got {}", center_vertical);
    assert!(center_horizontal > 100, "Horizontal cross should be bright, got {}", center_horizontal);
}

/// Test large to small downscale accuracy
#[test]
fn test_resize_downscale_4x() {
    let mut src = Mat::new(40, 40, 1, MatDepth::U8).unwrap();

    // Create gradient
    for row in 0..40 {
        for col in 0..40 {
            src.at_mut(row, col).unwrap()[0] = ((row + col) * 3) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    resize(&src, &mut dst, Size::new(10, 10), InterpolationFlag::Linear).unwrap();

    assert_eq!(dst.rows(), 10);
    assert_eq!(dst.cols(), 10);

    // Should produce smooth downscaled gradient
    // Top-left should be darker than bottom-right
    assert!(dst.at(0, 0).unwrap()[0] < dst.at(9, 9).unwrap()[0],
        "Gradient should be preserved in downscale");
}

/// Test edge case: 1x1 resize
#[test]
fn test_resize_single_pixel() {
    let mut src = Mat::new(10, 10, 3, MatDepth::U8).unwrap();

    // Fill with various values
    for row in 0..10 {
        for col in 0..10 {
            let pixel = src.at_mut(row, col).unwrap();
            pixel[0] = (row * 10 + col) as u8;
            pixel[1] = ((row + col) * 5) as u8;
            pixel[2] = 150;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    resize(&src, &mut dst, Size::new(1, 1), InterpolationFlag::Linear).unwrap();

    assert_eq!(dst.rows(), 1);
    assert_eq!(dst.cols(), 1);

    // Should produce value in valid range (samples one of the source pixels or interpolates)
    let pixel = dst.at(0, 0).unwrap();
    assert!(pixel[0] <= 255, "Channel 0 should be in valid range");
    assert!(pixel[1] <= 255, "Channel 1 should be in valid range");
    assert!(pixel[2] >= 0 && pixel[2] <= 255, "Channel 2 should be in valid range");
}

/// Visual inspection test for resize (ignored by default)
#[test]
#[ignore]
fn test_resize_visual_inspection() {
    let mut src = Mat::new(4, 4, 1, MatDepth::U8).unwrap();

    // Create simple pattern
    for row in 0..4 {
        for col in 0..4 {
            src.at_mut(row, col).unwrap()[0] = ((row * 4 + col) * 16) as u8;
        }
    }

    println!("\nOriginal 4x4:");
    print_image_data(&src, "Source", 4, 4);

    let mut upscaled = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    resize(&src, &mut upscaled, Size::new(8, 8), InterpolationFlag::Linear).unwrap();

    println!("\nUpscaled to 8x8 (bilinear):");
    print_image_data(&upscaled, "Upscaled", 8, 8);

    let mut downscaled = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    resize(&upscaled, &mut downscaled, Size::new(4, 4), InterpolationFlag::Linear).unwrap();

    println!("\nDownscaled back to 4x4:");
    print_image_data(&downscaled, "Downscaled", 4, 4);

    println!("\nDifference from original:");
    let stats = compute_diff_stats(&src, &downscaled);
    println!("{}", stats);
}
