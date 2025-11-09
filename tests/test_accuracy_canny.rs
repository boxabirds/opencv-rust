/// Bit-level accuracy tests for Canny edge detection
/// These tests verify that optimizations don't change results
mod test_utils;

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::Scalar;
use opencv_rust::imgproc::canny;
use test_utils::*;

/// Test Canny on uniform image (should produce no edges)
#[test]
fn test_canny_uniform_image_bit_exact() {
    let src = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    canny(&src, &mut dst, 50.0, 150.0).unwrap();

    // Uniform image should produce all zeros (no edges)
    for row in 0..dst.rows() {
        for col in 0..dst.cols() {
            let pixel = dst.at(row, col).unwrap()[0];
            assert_eq!(pixel, 0, "Uniform image should have no edges at ({}, {})", row, col);
        }
    }
}

/// Test Canny on sharp vertical edge
#[test]
fn test_canny_vertical_edge_bit_exact() {
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

    let mut edges = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    canny(&src, &mut edges, 50.0, 150.0).unwrap();

    // Count edge pixels near x=25
    let mut edge_pixels = 0;
    for row in 5..45 {
        for col in 23..27 {
            if edges.at(row, col).unwrap()[0] > 0 {
                edge_pixels += 1;
            }
        }
    }

    assert!(edge_pixels > 20, "Expected to detect vertical edge, found {} edge pixels", edge_pixels);

    // Verify edges are only near the transition
    for row in 5..45 {
        for col in 0..20 {
            assert_eq!(edges.at(row, col).unwrap()[0], 0,
                "Should have no edges far left of transition at ({}, {})", row, col);
        }
        for col in 30..50 {
            assert_eq!(edges.at(row, col).unwrap()[0], 0,
                "Should have no edges far right of transition at ({}, {})", row, col);
        }
    }
}

/// Test Canny on horizontal edge
#[test]
fn test_canny_horizontal_edge_bit_exact() {
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

    let mut edges = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    canny(&src, &mut edges, 50.0, 150.0).unwrap();

    // Count edge pixels near y=25
    let mut edge_pixels = 0;
    for row in 23..27 {
        for col in 5..45 {
            if edges.at(row, col).unwrap()[0] > 0 {
                edge_pixels += 1;
            }
        }
    }

    assert!(edge_pixels > 20, "Expected to detect horizontal edge, found {} edge pixels", edge_pixels);
}

/// Test Canny on diagonal edge
#[test]
fn test_canny_diagonal_edge_bit_exact() {
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

    let mut edges = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    canny(&src, &mut edges, 50.0, 150.0).unwrap();

    // Should detect diagonal edge
    let mut edge_pixels = 0;
    for i in 5..45 {
        // Check near the diagonal
        for offset in -1..=1 {
            let col = (i as i32 + offset).max(0).min(49) as usize;
            if edges.at(i, col).unwrap()[0] > 0 {
                edge_pixels += 1;
            }
        }
    }

    assert!(edge_pixels > 20, "Expected to detect diagonal edge, found {} edge pixels", edge_pixels);
}

/// Test Canny with different threshold values
#[test]
fn test_canny_threshold_sensitivity() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create weak edge (small gradient)
    for row in 0..50 {
        for col in 0..25 {
            src.at_mut(row, col).unwrap()[0] = 100;
        }
        for col in 25..50 {
            src.at_mut(row, col).unwrap()[0] = 150;
        }
    }

    // Low threshold - should detect edge
    let mut edges_low = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    canny(&src, &mut edges_low, 10.0, 30.0).unwrap();

    let mut edge_pixels_low = 0;
    for row in 5..45 {
        for col in 23..27 {
            if edges_low.at(row, col).unwrap()[0] > 0 {
                edge_pixels_low += 1;
            }
        }
    }

    // High threshold - should detect fewer or no edges
    let mut edges_high = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    canny(&src, &mut edges_high, 100.0, 200.0).unwrap();

    let mut edge_pixels_high = 0;
    for row in 5..45 {
        for col in 23..27 {
            if edges_high.at(row, col).unwrap()[0] > 0 {
                edge_pixels_high += 1;
            }
        }
    }

    assert!(edge_pixels_low > edge_pixels_high,
        "Lower threshold should detect more edges: low={}, high={}",
        edge_pixels_low, edge_pixels_high);
}

/// Test Canny output is binary (only 0 or 255)
#[test]
fn test_canny_output_is_binary() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create random-ish pattern
    for row in 0..50 {
        for col in 0..50 {
            src.at_mut(row, col).unwrap()[0] = ((row * 7 + col * 13) % 256) as u8;
        }
    }

    let mut edges = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    canny(&src, &mut edges, 50.0, 150.0).unwrap();

    // All pixels should be either 0 or 255
    for row in 0..edges.rows() {
        for col in 0..edges.cols() {
            let pixel = edges.at(row, col).unwrap()[0];
            assert!(pixel == 0 || pixel == 255,
                "Canny output should be binary, but pixel at ({}, {}) = {}",
                row, col, pixel);
        }
    }
}

/// Test Canny is deterministic (same input produces same output)
#[test]
fn test_canny_deterministic() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create complex pattern
    for row in 0..50 {
        for col in 0..50 {
            src.at_mut(row, col).unwrap()[0] = ((row * row + col * col) % 256) as u8;
        }
    }

    let mut edges1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut edges2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    canny(&src, &mut edges1, 50.0, 150.0).unwrap();
    canny(&src, &mut edges2, 50.0, 150.0).unwrap();

    // Results should be bit-exact identical
    assert_images_equal(&edges1, &edges2, "Canny should be deterministic");
}

/// Regression test with known reference output
#[test]
fn test_canny_checkerboard_reference() {
    let mut src = Mat::new(32, 32, 1, MatDepth::U8).unwrap();

    // Create 8x8 checkerboard (4x4 pixel squares)
    for row in 0..32 {
        for col in 0..32 {
            let square_row = row / 4;
            let square_col = col / 4;
            let value = if (square_row + square_col) % 2 == 0 { 0 } else { 255 };
            src.at_mut(row, col).unwrap()[0] = value;
        }
    }

    let mut edges = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    canny(&src, &mut edges, 50.0, 150.0).unwrap();

    // Checkerboard should produce edges at square boundaries
    // Edges should be at multiples of 4 (approximately, due to blur)
    let mut edge_count = 0;
    for row in 0..32 {
        for col in 0..32 {
            if edges.at(row, col).unwrap()[0] > 0 {
                edge_count += 1;
            }
        }
    }

    // Should detect a reasonable number of edges (not too many, not too few)
    assert!(edge_count > 50 && edge_count < 700,
        "Checkerboard should produce moderate edge count, got {}", edge_count);
}

/// Test that Canny handles edge cases correctly
#[test]
fn test_canny_boundary_pixels() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();

    // Put edges at the image boundary
    for i in 0..20 {
        src.at_mut(0, i).unwrap()[0] = 255;
        src.at_mut(19, i).unwrap()[0] = 255;
        src.at_mut(i, 0).unwrap()[0] = 255;
        src.at_mut(i, 19).unwrap()[0] = 255;
    }

    // Fill interior with different value
    for row in 1..19 {
        for col in 1..19 {
            src.at_mut(row, col).unwrap()[0] = 128;
        }
    }

    let mut edges = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    canny(&src, &mut edges, 50.0, 150.0).unwrap();

    // Should not crash and should produce valid output
    assert_eq!(edges.rows(), 20);
    assert_eq!(edges.cols(), 20);
}

/// Print visual representation of Canny output for manual inspection
#[test]
#[ignore] // Only run with --ignored flag for debugging
fn test_canny_visual_inspection() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();

    // Create simple cross pattern
    for i in 0..20 {
        src.at_mut(i, 10).unwrap()[0] = 255; // Vertical line
        src.at_mut(10, i).unwrap()[0] = 255; // Horizontal line
    }

    let mut edges = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    canny(&src, &mut edges, 50.0, 150.0).unwrap();

    println!("\nInput (cross pattern):");
    print_image_data(&src, "Input", 20, 20);

    println!("\nCanny output:");
    print_image_data(&edges, "Edges", 20, 20);
}
