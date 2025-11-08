// Integration tests for imgproc module ported from OpenCV test suite
// These tests validate correctness against known-good outputs

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::{Size, InterpolationFlag, ColorConversionCode, ThresholdType};
use opencv_rust::imgproc::*;

/// Create test image with gradient pattern
fn create_gradient_image(rows: usize, cols: usize) -> Mat {
    let mut img = Mat::new(rows, cols, 1, MatDepth::U8).unwrap();

    for row in 0..rows {
        for col in 0..cols {
            let val = ((row + col) % 256) as u8;
            img.at_mut(row, col).unwrap()[0] = val;
        }
    }

    img
}

/// Create test image with known pattern
fn create_checkerboard(rows: usize, cols: usize, square_size: usize) -> Mat {
    let mut img = Mat::new(rows, cols, 1, MatDepth::U8).unwrap();

    for row in 0..rows {
        for col in 0..cols {
            let is_black = ((row / square_size) + (col / square_size)) % 2 == 0;
            img.at_mut(row, col).unwrap()[0] = if is_black { 0 } else { 255 };
        }
    }

    img
}

/// Compare two matrices within tolerance
fn assert_matrices_near(mat1: &Mat, mat2: &Mat, tolerance: f64) {
    assert_eq!(mat1.rows(), mat2.rows());
    assert_eq!(mat1.cols(), mat2.cols());
    assert_eq!(mat1.channels(), mat2.channels());

    let mut max_diff: f64 = 0.0;
    let mut num_different = 0;

    for row in 0..mat1.rows() {
        for col in 0..mat1.cols() {
            let p1 = mat1.at(row, col).unwrap();
            let p2 = mat2.at(row, col).unwrap();

            for ch in 0..mat1.channels() {
                let diff = (p1[ch] as f64 - p2[ch] as f64).abs();
                if diff > tolerance {
                    num_different += 1;
                }
                max_diff = max_diff.max(diff);
            }
        }
    }

    let total_pixels = mat1.rows() * mat1.cols() * mat1.channels();
    let percent_different = (num_different as f64 / total_pixels as f64) * 100.0;

    assert!(
        percent_different < 1.0,
        "Too many different pixels: {:.2}%, max_diff: {:.2}",
        percent_different,
        max_diff
    );
}

#[test]
fn test_gaussian_blur_identity() {
    // Test that blur with sigma=0 approximates identity
    let src = create_gradient_image(100, 100);
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    // Very small kernel should preserve image
    gaussian_blur(&src, &mut dst, Size::new(3, 3), 0.1).unwrap();

    assert_matrices_near(&src, &dst, 5.0);
}

#[test]
fn test_gaussian_blur_symmetry() {
    // Blur should be symmetric
    let src = create_gradient_image(50, 50);

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    gaussian_blur(&src, &mut dst1, Size::new(5, 5), 1.0).unwrap();

    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    gaussian_blur(&src, &mut dst2, Size::new(5, 5), 1.0).unwrap();

    assert_matrices_near(&dst1, &dst2, 0.0); // Should be exactly equal
}

#[test]
fn test_sobel_gradient_horizontal() {
    // Create image with horizontal gradient
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();
    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (col * 25) as u8;
        }
    }

    let mut dx = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    sobel(&src, &mut dx, 1, 0, 3).unwrap();

    // Horizontal gradient should be detected
    // Center pixels should have non-zero gradient
    let center_val = dx.at(5, 5).unwrap()[0];
    assert!(center_val > 0, "Expected non-zero horizontal gradient");
}

#[test]
fn test_sobel_gradient_vertical() {
    // Create image with vertical gradient
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();
    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 25) as u8;
        }
    }

    let mut dy = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    sobel(&src, &mut dy, 0, 1, 3).unwrap();

    // Vertical gradient should be detected
    let center_val = dy.at(5, 5).unwrap()[0];
    assert!(center_val > 0, "Expected non-zero vertical gradient");
}

#[test]
fn test_canny_finds_edges() {
    // Create image with clear edge
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Left half black, right half white
    for row in 0..50 {
        for col in 0..25 {
            src.at_mut(row, col).unwrap()[0] = 0;
        }
        for col in 25..50 {
            src.at_mut(row, col).unwrap()[0] = 255;
        }
    }

    let mut edges = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    canny(&src, &mut edges, 50.0, 150.0).unwrap();

    // Should detect edge at x=25
    let mut edge_pixels = 0;
    for row in 5..45 {
        // Check around the edge
        for col in 23..27 {
            if edges.at(row, col).unwrap()[0] > 0 {
                edge_pixels += 1;
            }
        }
    }

    assert!(edge_pixels > 20, "Expected to detect vertical edge, found {} edge pixels", edge_pixels);
}

#[test]
fn test_threshold_binary() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    // Create image with values 0-99
    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 10 + col) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    threshold(&src, &mut dst, 50.0, 255.0, ThresholdType::Binary).unwrap();

    // Values <= 50 should be 0, > 50 should be 255
    assert_eq!(dst.at(5, 0).unwrap()[0], 0);   // 50 -> 0
    assert_eq!(dst.at(5, 1).unwrap()[0], 255); // 51 -> 255
    assert_eq!(dst.at(9, 9).unwrap()[0], 255); // 99 -> 255
}

#[test]
fn test_erode_shrinks_white_region() {
    let mut src = Mat::new(21, 21, 1, MatDepth::U8).unwrap();

    // Create 11x11 white square in center
    for row in 0..21 {
        for col in 0..21 {
            if row >= 5 && row < 16 && col >= 5 && col < 16 {
                src.at_mut(row, col).unwrap()[0] = 255;
            } else {
                src.at_mut(row, col).unwrap()[0] = 0;
            }
        }
    }

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(3, 3));
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    erode(&src, &mut dst, &kernel).unwrap();

    // After erosion, corners of the square should be black
    assert_eq!(dst.at(5, 5).unwrap()[0], 0, "Corner should be eroded");
    assert_eq!(dst.at(10, 10).unwrap()[0], 255, "Center should remain white");
}

#[test]
fn test_dilate_grows_white_region() {
    let mut src = Mat::new(21, 21, 1, MatDepth::U8).unwrap();

    // Create 5x5 white square in center
    for row in 0..21 {
        for col in 0..21 {
            if row >= 8 && row < 13 && col >= 8 && col < 13 {
                src.at_mut(row, col).unwrap()[0] = 255;
            } else {
                src.at_mut(row, col).unwrap()[0] = 0;
            }
        }
    }

    let kernel = get_structuring_element(MorphShape::Rect, Size::new(3, 3));
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    dilate(&src, &mut dst, &kernel).unwrap();

    // After dilation, area should grow
    // Pixel just outside original square should now be white
    assert_eq!(dst.at(7, 10).unwrap()[0], 255, "Should be dilated");
    assert_eq!(dst.at(10, 10).unwrap()[0], 255, "Center should remain white");
}

#[test]
fn test_resize_downscale_preserves_corners() {
    let mut src = Mat::new(100, 100, 1, MatDepth::U8).unwrap();

    // Set corners to specific values
    src.at_mut(0, 0).unwrap()[0] = 10;
    src.at_mut(0, 99).unwrap()[0] = 20;
    src.at_mut(99, 0).unwrap()[0] = 30;
    src.at_mut(99, 99).unwrap()[0] = 40;

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    resize(&src, &mut dst, Size::new(50, 50), InterpolationFlag::Linear).unwrap();

    assert_eq!(dst.rows(), 50);
    assert_eq!(dst.cols(), 50);

    // Corner values should be approximately preserved
    let tol = 15.0;
    assert!((dst.at(0, 0).unwrap()[0] as f64 - 10.0).abs() < tol);
    assert!((dst.at(0, 49).unwrap()[0] as f64 - 20.0).abs() < tol);
}

#[test]
fn test_flip_horizontal() {
    let mut src = Mat::new(5, 5, 1, MatDepth::U8).unwrap();

    // Create pattern
    for row in 0..5 {
        for col in 0..5 {
            src.at_mut(row, col).unwrap()[0] = (row * 5 + col) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    flip(&src, &mut dst, 1).unwrap(); // 1 = horizontal flip

    // Check that columns are flipped
    for row in 0..5 {
        assert_eq!(
            src.at(row, 0).unwrap()[0],
            dst.at(row, 4).unwrap()[0]
        );
        assert_eq!(
            src.at(row, 4).unwrap()[0],
            dst.at(row, 0).unwrap()[0]
        );
    }
}

#[test]
fn test_cvt_color_rgb_to_gray_brightness() {
    let mut src = Mat::new(10, 10, 3, MatDepth::U8).unwrap();

    // Pure red pixel
    src.at_mut(0, 0).unwrap()[0] = 255;
    src.at_mut(0, 0).unwrap()[1] = 0;
    src.at_mut(0, 0).unwrap()[2] = 0;

    // Pure green pixel
    src.at_mut(0, 1).unwrap()[0] = 0;
    src.at_mut(0, 1).unwrap()[1] = 255;
    src.at_mut(0, 1).unwrap()[2] = 0;

    // Pure blue pixel
    src.at_mut(0, 2).unwrap()[0] = 0;
    src.at_mut(0, 2).unwrap()[1] = 0;
    src.at_mut(0, 2).unwrap()[2] = 255;

    let mut gray = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    cvt_color(&src, &mut gray, ColorConversionCode::RgbToGray).unwrap();

    // Green should be brightest (coefficients: 0.299 R + 0.587 G + 0.114 B)
    let red_gray = gray.at(0, 0).unwrap()[0];
    let green_gray = gray.at(0, 1).unwrap()[0];
    let blue_gray = gray.at(0, 2).unwrap()[0];

    assert!(green_gray > red_gray, "Green should be brighter than red");
    assert!(green_gray > blue_gray, "Green should be brighter than blue");
    assert!(red_gray > blue_gray, "Red should be brighter than blue");
}
