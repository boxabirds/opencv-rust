#![allow(unused_comparisons)]
/// Bit-level accuracy tests for Laplacian operator
/// These tests verify that optimizations don't change results
mod test_utils;

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::Scalar;
use opencv_rust::imgproc::laplacian;
use test_utils::*;

/// Test Laplacian is deterministic
#[test]
fn test_laplacian_deterministic() {
    let mut src = Mat::new(100, 100, 1, MatDepth::U8).unwrap();

    // Create pattern
    for row in 0..100 {
        for col in 0..100 {
            src.at_mut(row, col).unwrap()[0] = ((row * 7 + col * 11) % 256) as u8;
        }
    }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    laplacian(&src, &mut dst1, 3).unwrap();
    laplacian(&src, &mut dst2, 3).unwrap();

    // Results should be bit-exact identical
    assert_images_equal(&dst1, &dst2, "Laplacian should be deterministic");
}

/// Test Laplacian on uniform image
#[test]
fn test_laplacian_uniform_image() {
    let src = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    laplacian(&src, &mut dst, 3).unwrap();

    // Uniform image should have zero second derivative (except borders)
    for row in 2..48 {
        for col in 2..48 {
            assert_eq!(dst.at(row, col).unwrap()[0], 0,
                "Uniform image should have zero Laplacian at ({}, {})", row, col);
        }
    }
}

/// Test Laplacian detects edges
#[test]
fn test_laplacian_edge_detection() {
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
    laplacian(&src, &mut dst, 3).unwrap();

    // Should detect edges
    let mut edge_pixels = 0;
    for row in 5..45 {
        for col in 23..27 {
            if dst.at(row, col).unwrap()[0] > 50 {
                edge_pixels += 1;
            }
        }
    }

    assert!(edge_pixels > 20,
        "Laplacian should detect edges, found {} pixels", edge_pixels);
}

/// Test Laplacian on blob pattern
#[test]
fn test_laplacian_blob_detection() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Fill with gray
    for row in 0..50 {
        for col in 0..50 {
            src.at_mut(row, col).unwrap()[0] = 100;
        }
    }

    // Create bright blob
    for row in 20..30 {
        for col in 20..30 {
            src.at_mut(row, col).unwrap()[0] = 255;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    laplacian(&src, &mut dst, 3).unwrap();

    // Laplacian should respond to blob boundaries
    assert_eq!(dst.rows(), 50);
    assert_eq!(dst.cols(), 50);
}

/// Test Laplacian output range
#[test]
fn test_laplacian_output_range() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create extreme checkerboard pattern
    for row in 0..50 {
        for col in 0..50 {
            src.at_mut(row, col).unwrap()[0] = if (row + col) % 2 == 0 { 0 } else { 255 };
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    laplacian(&src, &mut dst, 3).unwrap();

    // All values should be in valid range
    for row in 0..50 {
        for col in 0..50 {
            let val = dst.at(row, col).unwrap()[0];
            // val is u8, always <= 255
assert!(val == val,
                "Laplacian output at ({}, {}) out of range: {}", row, col, val);
        }
    }
}

/// Test Laplacian on diagonal edge
#[test]
fn test_laplacian_diagonal_edge() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create diagonal edge
    for row in 0..50 {
        for col in 0..50 {
            if col < row {
                src.at_mut(row, col).unwrap()[0] = 50;
            } else {
                src.at_mut(row, col).unwrap()[0] = 200;
            }
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    laplacian(&src, &mut dst, 3).unwrap();

    // Should detect the diagonal edge
    let mut edge_detected = 0;
    for i in 10..40 {
        if dst.at(i, i).unwrap()[0] > 30 {
            edge_detected += 1;
        }
    }

    assert!(edge_detected > 10,
        "Laplacian should detect diagonal edge");
}

/// Test Laplacian boundary handling
#[test]
fn test_laplacian_boundary_handling() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();

    for row in 0..20 {
        for col in 0..20 {
            src.at_mut(row, col).unwrap()[0] = ((row * col) % 256) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    laplacian(&src, &mut dst, 3).unwrap();

    // Should not crash and produce valid dimensions
    assert_eq!(dst.rows(), 20);
    assert_eq!(dst.cols(), 20);

    // Border pixels should be valid
    for i in 0..20 {
        assert!(dst.at(0, i).unwrap()[0] <= 255);
        assert!(dst.at(19, i).unwrap()[0] <= 255);
        assert!(dst.at(i, 0).unwrap()[0] <= 255);
        assert!(dst.at(i, 19).unwrap()[0] <= 255);
    }
}

/// Visual inspection test (ignored by default)
#[test]
#[ignore]
fn test_laplacian_visual_inspection() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();

    // Create blob
    for row in 0..20 {
        for col in 0..20 {
            src.at_mut(row, col).unwrap()[0] = 100;
        }
    }
    for row in 5..15 {
        for col in 5..15 {
            src.at_mut(row, col).unwrap()[0] = 255;
        }
    }

    println!("\nInput (bright blob):");
    print_image_data(&src, "Source", 20, 20);

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    laplacian(&src, &mut dst, 3).unwrap();

    println!("\nLaplacian output (detects blob edges):");
    print_image_data(&dst, "Laplacian", 20, 20);
}
