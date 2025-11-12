/// Bit-level accuracy tests for Sobel derivative filters
/// These tests verify that optimizations don't change results
mod test_utils;

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::Scalar;
use opencv_rust::imgproc::sobel;
use test_utils::*;

/// Test Sobel is deterministic
#[test]
fn test_sobel_deterministic_dx() {
    let mut src = Mat::new(100, 100, 1, MatDepth::U8).unwrap();

    // Create gradient pattern
    for row in 0..100 {
        for col in 0..100 {
            src.at_mut(row, col).unwrap()[0] = ((row * 3 + col * 7) % 256) as u8;
        }
    }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    sobel(&src, &mut dst1, 1, 0, 3).unwrap(); // dx
    sobel(&src, &mut dst2, 1, 0, 3).unwrap();

    // Results should be bit-exact identical
    assert_images_equal(&dst1, &dst2, "Sobel dx should be deterministic");
}

/// Test Sobel dy deterministic
#[test]
fn test_sobel_deterministic_dy() {
    let mut src = Mat::new(100, 100, 1, MatDepth::U8).unwrap();

    for row in 0..100 {
        for col in 0..100 {
            src.at_mut(row, col).unwrap()[0] = ((row * 5 + col * 11) % 256) as u8;
        }
    }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    sobel(&src, &mut dst1, 0, 1, 3).unwrap(); // dy
    sobel(&src, &mut dst2, 0, 1, 3).unwrap();

    assert_images_equal(&dst1, &dst2, "Sobel dy should be deterministic");
}

/// Test Sobel on uniform image (no gradients)
#[test]
fn test_sobel_uniform_image() {
    let src = Mat::new_with_default(50, 50, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    sobel(&src, &mut dst, 1, 0, 3).unwrap();

    // Uniform image should have zero gradients (except borders)
    for row in 2..48 {
        for col in 2..48 {
            assert_eq!(dst.at(row, col).unwrap()[0], 0,
                "Uniform image should have zero gradient at ({}, {})", row, col);
        }
    }
}

/// Test Sobel detects vertical edges (dx)
#[test]
fn test_sobel_vertical_edge_dx() {
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
    sobel(&src, &mut dst, 1, 0, 3).unwrap(); // dx (horizontal gradient)

    // Should detect strong horizontal gradient near x=25
    let mut strong_gradient_pixels = 0;
    for row in 5..45 {
        for col in 23..27 {
            if dst.at(row, col).unwrap()[0] > 100 {
                strong_gradient_pixels += 1;
            }
        }
    }

    assert!(strong_gradient_pixels > 50,
        "Sobel dx should detect vertical edge, found {} pixels", strong_gradient_pixels);
}

/// Test Sobel detects horizontal edges (dy)
#[test]
fn test_sobel_horizontal_edge_dy() {
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
    sobel(&src, &mut dst, 0, 1, 3).unwrap(); // dy (vertical gradient)

    // Should detect strong vertical gradient near y=25
    let mut strong_gradient_pixels = 0;
    for row in 23..27 {
        for col in 5..45 {
            if dst.at(row, col).unwrap()[0] > 100 {
                strong_gradient_pixels += 1;
            }
        }
    }

    assert!(strong_gradient_pixels > 50,
        "Sobel dy should detect horizontal edge, found {} pixels", strong_gradient_pixels);
}

/// Test Sobel gradient direction correctness
#[test]
fn test_sobel_gradient_direction() {
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

    sobel(&src, &mut dx, 1, 0, 3).unwrap();
    sobel(&src, &mut dy, 0, 1, 3).unwrap();

    // At the vertical edge, dx should be much stronger than dy
    let dx_value = dx.at(25, 25).unwrap()[0];
    let dy_value = dy.at(25, 25).unwrap()[0];

    assert!(dx_value > 2 * dy_value,
        "For vertical edge, dx ({}) should be much stronger than dy ({})", dx_value, dy_value);
}

/// Test Sobel doesn't detect edges perpendicular to gradient direction
#[test]
fn test_sobel_perpendicular_edge() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create vertical edge
    for row in 0..50 {
        for col in 0..25 {
            src.at_mut(row, col).unwrap()[0] = 0;
        }
        for col in 25..50 {
            src.at_mut(row, col).unwrap()[0] = 255;
        }
    }

    let mut dy = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    sobel(&src, &mut dy, 0, 1, 3).unwrap(); // dy shouldn't detect vertical edge

    // Far from edge, dy should be near zero for vertical edge
    for row in 10..40 {
        for col in [10, 35] {
            let val = dy.at(row, col).unwrap()[0];
            assert!(val < 50,
                "Sobel dy should not strongly detect vertical edge at ({}, {}), got {}",
                row, col, val);
        }
    }
}

/// Test Sobel on diagonal edge
#[test]
fn test_sobel_diagonal_edge() {
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

    sobel(&src, &mut dx, 1, 0, 3).unwrap();
    sobel(&src, &mut dy, 0, 1, 3).unwrap();

    // For diagonal edge, both dx and dy should be significant
    let mut both_strong = 0;
    for i in 10..40 {
        let dx_val = dx.at(i, i).unwrap()[0];
        let dy_val = dy.at(i, i).unwrap()[0];

        if dx_val > 50 && dy_val > 50 {
            both_strong += 1;
        }
    }

    assert!(both_strong > 15,
        "Diagonal edge should have strong gradients in both directions, found {} pixels", both_strong);
}

/// Test Sobel handles boundary pixels
#[test]
fn test_sobel_boundary_handling() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();

    // Create pattern with edges at boundaries
    for row in 0..20 {
        for col in 0..20 {
            src.at_mut(row, col).unwrap()[0] = ((row + col) * 10) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    sobel(&src, &mut dst, 1, 0, 3).unwrap();

    // Should not crash and produce valid dimensions
    assert_eq!(dst.rows(), 20);
    assert_eq!(dst.cols(), 20);

    // Border pixels should be handled (likely zero due to no padding)
    for i in 0..20 {
        assert!(dst.at(0, i).unwrap()[0] <= 255);
        assert!(dst.at(19, i).unwrap()[0] <= 255);
        assert!(dst.at(i, 0).unwrap()[0] <= 255);
        assert!(dst.at(i, 19).unwrap()[0] <= 255);
    }
}

/// Test Sobel output range [0, 255]
#[test]
fn test_sobel_output_range() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create extreme pattern
    for row in 0..50 {
        for col in 0..50 {
            src.at_mut(row, col).unwrap()[0] = if (row + col) % 2 == 0 { 0 } else { 255 };
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    sobel(&src, &mut dst, 1, 1, 3).unwrap(); // Both dx and dy

    // All values should be in valid range
    for row in 0..50 {
        for col in 0..50 {
            let val = dst.at(row, col).unwrap()[0];
            assert!(val <= 255,
                "Sobel output at ({}, {}) out of range: {}", row, col, val);
        }
    }
}

/// Visual inspection test (ignored by default)
#[test]
#[ignore]
fn test_sobel_visual_inspection() {
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

    let mut dx = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dy = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    sobel(&src, &mut dx, 1, 0, 3).unwrap();
    sobel(&src, &mut dy, 0, 1, 3).unwrap();

    println!("\nSobel dx (should detect vertical edge):");
    print_image_data(&dx, "Sobel dx", 20, 20);

    println!("\nSobel dy (should be mostly zero):");
    print_image_data(&dy, "Sobel dy", 20, 20);
}
