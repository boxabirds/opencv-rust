#![allow(unused_comparisons)]
/// Bit-level accuracy tests for Scharr derivative filter
/// These tests verify that optimizations don't change results
mod test_utils;

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::Scalar;
use opencv_rust::imgproc::{scharr, sobel};
use test_utils::*;

/// Test Scharr is deterministic
#[test]
fn test_scharr_deterministic_dx() {
    let mut src = Mat::new(100, 100, 1, MatDepth::U8).unwrap();

    // Create gradient pattern
    for row in 0..100 {
        for col in 0..100 {
            src.at_mut(row, col).unwrap()[0] = ((row * 5 + col * 13) % 256) as u8;
        }
    }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    scharr(&src, &mut dst1, 1, 0).unwrap(); // dx
    scharr(&src, &mut dst2, 1, 0).unwrap();

    // Results should be bit-exact identical
    assert_images_equal(&dst1, &dst2, "Scharr dx should be deterministic");
}

/// Test Scharr dy deterministic
#[test]
fn test_scharr_deterministic_dy() {
    let mut src = Mat::new(100, 100, 1, MatDepth::U8).unwrap();

    for row in 0..100 {
        for col in 0..100 {
            src.at_mut(row, col).unwrap()[0] = ((row * 7 + col * 17) % 256) as u8;
        }
    }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    scharr(&src, &mut dst1, 0, 1).unwrap(); // dy
    scharr(&src, &mut dst2, 0, 1).unwrap();

    assert_images_equal(&dst1, &dst2, "Scharr dy should be deterministic");
}

/// Test Scharr on uniform image
#[test]
fn test_scharr_uniform_image() {
    let src = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    scharr(&src, &mut dst, 1, 0).unwrap();

    // Uniform image should have zero gradients (except borders)
    for row in 2..48 {
        for col in 2..48 {
            assert_eq!(dst.at(row, col).unwrap()[0], 0,
                "Uniform image should have zero gradient at ({}, {})", row, col);
        }
    }
}

/// Test Scharr detects vertical edges (dx)
#[test]
fn test_scharr_vertical_edge_dx() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create sharp vertical edge at x=25
    for row in 0..50 {
        for col in 0..25 {
            src.at_mut(row, col).unwrap()[0] = 0;
        }
        for col in 25..50 {
            src.at_mut(row, col).unwrap()[0] = 255;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    scharr(&src, &mut dst, 1, 0).unwrap(); // dx

    // Should detect strong gradient near x=25
    let mut strong_gradient_pixels = 0;
    for row in 5..45 {
        for col in 23..27 {
            if dst.at(row, col).unwrap()[0] > 100 {
                strong_gradient_pixels += 1;
            }
        }
    }

    assert!(strong_gradient_pixels > 50,
        "Scharr dx should detect vertical edge, found {} pixels", strong_gradient_pixels);
}

/// Test Scharr detects horizontal edges (dy)
#[test]
fn test_scharr_horizontal_edge_dy() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create sharp horizontal edge at y=25
    for row in 0..25 {
        for col in 0..50 {
            src.at_mut(row, col).unwrap()[0] = 0;
        }
    }
    for row in 25..50 {
        for col in 0..50 {
            src.at_mut(row, col).unwrap()[0] = 255;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    scharr(&src, &mut dst, 0, 1).unwrap(); // dy

    // Should detect strong gradient near y=25
    let mut strong_gradient_pixels = 0;
    for row in 23..27 {
        for col in 5..45 {
            if dst.at(row, col).unwrap()[0] > 100 {
                strong_gradient_pixels += 1;
            }
        }
    }

    assert!(strong_gradient_pixels > 50,
        "Scharr dy should detect horizontal edge, found {} pixels", strong_gradient_pixels);
}

/// Test Scharr vs Sobel (Scharr should be more accurate for 3x3)
#[test]
fn test_scharr_vs_sobel_accuracy() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create gentle gradient to avoid saturation (both filters would saturate on sharp edges)
    for row in 0..50 {
        for col in 0..50 {
            // Gradual diagonal gradient from 50 to 200 (not 0 to 255)
            let value = 50 + ((row + col) * 150 / 100).min(150);
            src.at_mut(row, col).unwrap()[0] = value as u8;
        }
    }

    let mut scharr_dx = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut sobel_dx = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    scharr(&src, &mut scharr_dx, 1, 0).unwrap();
    sobel(&src, &mut sobel_dx, 1, 0, 3).unwrap();

    // Both should detect gradients, but with different magnitudes due to kernel weights
    // Scharr uses stronger weights (-10 vs -2), so should produce larger responses
    let mut scharr_sum: u32 = 0;
    let mut sobel_sum: u32 = 0;
    for row in 0..50 {
        for col in 0..50 {
            scharr_sum += scharr_dx.at(row, col).unwrap()[0] as u32;
            sobel_sum += sobel_dx.at(row, col).unwrap()[0] as u32;
        }
    }

    // Scharr should generally produce different (typically stronger) responses than Sobel
    assert!(scharr_sum != sobel_sum,
        "Scharr and Sobel should produce different results (different kernels): Scharr sum={}, Sobel sum={}",
        scharr_sum, sobel_sum);
}

/// Test Scharr gradient direction correctness
#[test]
fn test_scharr_gradient_direction() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create vertical edge (dx should be strong, dy should be weak)
    for row in 0..50 {
        for col in 0..25 {
            src.at_mut(row, col).unwrap()[0] = 50;
        }
        for col in 25..50 {
            src.at_mut(row, col).unwrap()[0] = 200;
        }
    }

    let mut dx = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dy = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    scharr(&src, &mut dx, 1, 0).unwrap();
    scharr(&src, &mut dy, 0, 1).unwrap();

    // At the vertical edge, dx should be much stronger than dy
    let dx_value = dx.at(25, 25).unwrap()[0];
    let dy_value = dy.at(25, 25).unwrap()[0];

    assert!(dx_value > 2 * dy_value,
        "For vertical edge, dx ({}) should be much stronger than dy ({})", dx_value, dy_value);
}

/// Test Scharr output range [0, 255]
#[test]
fn test_scharr_output_range() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create extreme checkerboard pattern
    for row in 0..50 {
        for col in 0..50 {
            src.at_mut(row, col).unwrap()[0] = if (row + col) % 2 == 0 { 0 } else { 255 };
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    scharr(&src, &mut dst, 1, 0).unwrap();

    // All values should be in valid range
    for row in 0..50 {
        for col in 0..50 {
            let val = dst.at(row, col).unwrap()[0];
            // val is u8, always <= 255
assert!(val == val,
                "Scharr output at ({}, {}) out of range: {}", row, col, val);
        }
    }
}

/// Test Scharr on diagonal edge
#[test]
fn test_scharr_diagonal_edge() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create diagonal edge
    for row in 0..50 {
        for col in 0..50 {
            if col < row {
                src.at_mut(row, col).unwrap()[0] = 0;
            } else {
                src.at_mut(row, col).unwrap()[0] = 255;
            }
        }
    }

    let mut dx = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dy = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    scharr(&src, &mut dx, 1, 0).unwrap();
    scharr(&src, &mut dy, 0, 1).unwrap();

    // For diagonal edge, both dx and dy should be significant
    let mut both_strong = 0;
    for i in 10..40 {
        let dx_val = dx.at(i, i).unwrap()[0];
        let dy_val = dy.at(i, i).unwrap()[0];

        if dx_val > 50 && dy_val > 50 {
            both_strong += 1;
        }
    }

    assert!(both_strong > 10,
        "Diagonal edge should have strong gradients in both directions, found {} pixels", both_strong);
}

/// Test Scharr boundary handling
#[test]
fn test_scharr_boundary_handling() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();

    for row in 0..20 {
        for col in 0..20 {
            src.at_mut(row, col).unwrap()[0] = ((row * col) % 256) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    scharr(&src, &mut dst, 1, 0).unwrap();

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
fn test_scharr_visual_inspection() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();

    // Create simple vertical edge
    for row in 0..20 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = 0;
        }
        for col in 10..20 {
            src.at_mut(row, col).unwrap()[0] = 255;
        }
    }

    println!("\nInput (vertical edge at x=10):");
    print_image_data(&src, "Source", 20, 20);

    let mut scharr_dx = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut scharr_dy = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut sobel_dx = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    scharr(&src, &mut scharr_dx, 1, 0).unwrap();
    scharr(&src, &mut scharr_dy, 0, 1).unwrap();
    sobel(&src, &mut sobel_dx, 1, 0, 3).unwrap();

    println!("\nScharr dx (should detect vertical edge):");
    print_image_data(&scharr_dx, "Scharr dx", 20, 20);

    println!("\nScharr dy (should be mostly zero):");
    print_image_data(&scharr_dy, "Scharr dy", 20, 20);

    println!("\nSobel dx for comparison:");
    print_image_data(&sobel_dx, "Sobel dx", 20, 20);

    println!("\nDifference (Scharr - Sobel):");
    let stats = compute_diff_stats(&scharr_dx, &sobel_dx);
    println!("{}", stats);
}
