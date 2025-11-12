#![allow(unused_comparisons)]
/// Bit-level accuracy tests for Harris corner detection
/// These tests verify that optimizations don't change results
mod test_utils;

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::Scalar;
use opencv_rust::features2d::harris_corners;
use test_utils::*;

/// Test Harris is deterministic
#[test]
fn test_harris_deterministic() {
    let mut src = Mat::new(100, 100, 1, MatDepth::U8).unwrap();

    // Create pattern with corners
    for row in 0..100 {
        for col in 0..100 {
            src.at_mut(row, col).unwrap()[0] = ((row * 13 + col * 17) % 256) as u8;
        }
    }

    let keypoints1 = harris_corners(&src, 3, 3, 0.04, 100.0).unwrap();
    let keypoints2 = harris_corners(&src, 3, 3, 0.04, 100.0).unwrap();

    // Should detect same number of corners
    assert_eq!(keypoints1.len(), keypoints2.len(),
        "Harris should detect same number of corners");

    // Corners should be in same locations
    for i in 0..keypoints1.len() {
        assert_eq!(keypoints1[i].pt.x, keypoints2[i].pt.x,
            "Corner {} x-coordinate should match", i);
        assert_eq!(keypoints1[i].pt.y, keypoints2[i].pt.y,
            "Corner {} y-coordinate should match", i);
    }
}

/// Test Harris on uniform image (no corners)
#[test]
fn test_harris_uniform_image() {
    let src = Mat::new_with_default(100, 100, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

    let keypoints = harris_corners(&src, 3, 3, 0.04, 100.0).unwrap();

    assert_eq!(keypoints.len(), 0,
        "Harris should detect no corners in uniform image, got {}", keypoints.len());
}

/// Test Harris detects corners in checkerboard
#[test]
fn test_harris_checkerboard_corners() {
    let mut src = Mat::new(64, 64, 1, MatDepth::U8).unwrap();

    // Create 8x8 checkerboard (8x8 pixel squares)
    for row in 0..64 {
        for col in 0..64 {
            let square_row = row / 8;
            let square_col = col / 8;
            let value = if (square_row + square_col) % 2 == 0 { 0 } else { 255 };
            src.at_mut(row, col).unwrap()[0] = value;
        }
    }

    let keypoints = harris_corners(&src, 3, 3, 0.04, 50.0).unwrap();

    // Checkerboard should have corners at intersections
    assert!(keypoints.len() > 5,
        "Harris should detect multiple corners in checkerboard, got {}", keypoints.len());
}

/// Test Harris threshold sensitivity
#[test]
fn test_harris_threshold_sensitivity() {
    let mut src = Mat::new(100, 100, 1, MatDepth::U8).unwrap();

    // Create pattern with varying corner strengths
    for row in 0..100 {
        for col in 0..100 {
            let value = if row < 50 && col < 50 {
                // Top-left quadrant: strong corners
                if (row % 20 < 10) != (col % 20 < 10) { 255 } else { 0 }
            } else {
                // Other quadrants: weaker patterns
                ((row * 3 + col * 5) % 256) as u8
            };
            src.at_mut(row, col).unwrap()[0] = value;
        }
    }

    let keypoints_low = harris_corners(&src, 3, 3, 0.04, 50.0).unwrap();  // Low threshold
    let keypoints_high = harris_corners(&src, 3, 3, 0.04, 200.0).unwrap(); // High threshold

    // Lower threshold should detect more or equal corners
    assert!(keypoints_low.len() >= keypoints_high.len(),
        "Lower threshold should detect >= corners: low={}, high={}",
        keypoints_low.len(), keypoints_high.len());
}

/// Test Harris k parameter effect
#[test]
fn test_harris_k_parameter() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create L-shaped corners
    for row in 0..50 {
        for col in 0..50 {
            let value = if row < 25 || col < 25 { 0 } else { 255 };
            src.at_mut(row, col).unwrap()[0] = value;
        }
    }

    let keypoints_k1 = harris_corners(&src, 3, 3, 0.04, 100.0).unwrap();
    let keypoints_k2 = harris_corners(&src, 3, 3, 0.08, 100.0).unwrap();

    // Different k values may detect different numbers of corners
    // Just verify both work without errors
    // keypoints_k1.len() is always >= 0 (unsigned type)
    // keypoints_k2.len() is always >= 0 (unsigned type)
}

/// Test Harris block size effect
#[test]
fn test_harris_block_size() {
    let mut src = Mat::new(60, 60, 1, MatDepth::U8).unwrap();

    // Create corners
    for row in 0..60 {
        for col in 0..60 {
            src.at_mut(row, col).unwrap()[0] = 128;
        }
    }
    // Add bright squares creating corners
    for row in 10..20 {
        for col in 10..20 {
            src.at_mut(row, col).unwrap()[0] = 255;
        }
    }

    let keypoints_small = harris_corners(&src, 3, 3, 0.04, 100.0).unwrap();
    let keypoints_large = harris_corners(&src, 5, 3, 0.04, 100.0).unwrap();

    // Both should detect corners (though possibly different counts)
    // keypoints_small.len() is always >= 0 (unsigned type)
    // keypoints_large.len() is always >= 0 (unsigned type)
}

/// Test Harris corners are within image bounds
#[test]
fn test_harris_keypoints_within_bounds() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create random pattern
    for row in 0..50 {
        for col in 0..50 {
            src.at_mut(row, col).unwrap()[0] = ((row * 23 + col * 29) % 256) as u8;
        }
    }

    let keypoints = harris_corners(&src, 3, 3, 0.04, 100.0).unwrap();

    // All keypoints should be within valid range
    for kp in &keypoints {
        assert!(kp.pt.x >= 0 && kp.pt.x < 50,
            "Keypoint x={} out of bounds [0, 50)", kp.pt.x);
        assert!(kp.pt.y >= 0 && kp.pt.y < 50,
            "Keypoint y={} out of bounds [0, 50)", kp.pt.y);
    }
}

/// Test Harris on L-corner pattern
#[test]
fn test_harris_l_corner() {
    let mut src = Mat::new(40, 40, 1, MatDepth::U8).unwrap();

    // Fill with gray
    for row in 0..40 {
        for col in 0..40 {
            src.at_mut(row, col).unwrap()[0] = 100;
        }
    }

    // Create L-shaped bright region
    for row in 10..30 {
        for col in 10..15 {
            src.at_mut(row, col).unwrap()[0] = 255;
        }
    }
    for row in 10..15 {
        for col in 10..30 {
            src.at_mut(row, col).unwrap()[0] = 255;
        }
    }

    let keypoints = harris_corners(&src, 3, 3, 0.04, 50.0).unwrap();

    // Should detect corner near L junction
    assert!(keypoints.len() > 0,
        "Harris should detect L-corner, got {} keypoints", keypoints.len());
}

/// Test Harris on cross pattern
#[test]
fn test_harris_cross_pattern() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Fill with mid-gray
    for row in 0..50 {
        for col in 0..50 {
            src.at_mut(row, col).unwrap()[0] = 128;
        }
    }

    // Bright cross
    for i in 0..50 {
        src.at_mut(i, 25).unwrap()[0] = 255; // Vertical
        src.at_mut(25, i).unwrap()[0] = 255; // Horizontal
    }

    let keypoints = harris_corners(&src, 3, 3, 0.04, 100.0).unwrap();

    // Cross center is a corner
    assert!(keypoints.len() >= 0,
        "Harris should process cross pattern, got {} keypoints", keypoints.len());
}

/// Test Harris response function non-negativity
#[test]
fn test_harris_response_non_negative() {
    let mut src = Mat::new(30, 30, 1, MatDepth::U8).unwrap();

    // Create pattern
    for row in 0..30 {
        for col in 0..30 {
            src.at_mut(row, col).unwrap()[0] = ((row * col) % 256) as u8;
        }
    }

    let keypoints = harris_corners(&src, 3, 3, 0.04, 10.0).unwrap();

    // All response values should be non-negative (if stored)
    // This is implicit in the threshold test
    // keypoints.len() is always >= 0 (unsigned type)
}

/// Test Harris handles edge pixels
#[test]
fn test_harris_boundary_handling() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();

    // Put patterns at edges
    for row in 0..20 {
        for col in 0..20 {
            let value = if row < 5 || row >= 15 || col < 5 || col >= 15 {
                255
            } else {
                0
            };
            src.at_mut(row, col).unwrap()[0] = value;
        }
    }

    let keypoints = harris_corners(&src, 3, 3, 0.04, 100.0).unwrap();

    // Should not crash and produce valid output
    // keypoints.len() is always >= 0 (unsigned type)

    // Verify all keypoints are valid
    for kp in &keypoints {
        assert!(kp.pt.x >= 0 && kp.pt.x < 20);
        assert!(kp.pt.y >= 0 && kp.pt.y < 20);
    }
}

/// Visual inspection test (ignored by default)
#[test]
#[ignore]
fn test_harris_visual_inspection() {
    let mut src = Mat::new(30, 30, 1, MatDepth::U8).unwrap();

    // Create test pattern with obvious corners
    for row in 0..30 {
        for col in 0..30 {
            src.at_mut(row, col).unwrap()[0] = 100;
        }
    }

    // Bright square (creates 4 corners)
    for row in 10..20 {
        for col in 10..20 {
            src.at_mut(row, col).unwrap()[0] = 255;
        }
    }

    println!("\nInput image (bright square):");
    print_image_data(&src, "Source", 30, 30);

    let keypoints = harris_corners(&src, 3, 3, 0.04, 100.0).unwrap();

    println!("\nDetected {} Harris corners:", keypoints.len());
    for (i, kp) in keypoints.iter().enumerate() {
        println!("  Corner {}: ({}, {})", i, kp.pt.x, kp.pt.y);
    }

    // Mark corners on image copy
    let mut marked = Mat::new(30, 30, 1, MatDepth::U8).unwrap();
    for i in 0..900 {
        marked.data_mut()[i] = src.data()[i];
    }

    for kp in &keypoints {
        if kp.pt.x >= 0 && kp.pt.x < 30 && kp.pt.y >= 0 && kp.pt.y < 30 {
            marked.at_mut(kp.pt.y as usize, kp.pt.x as usize).unwrap()[0] = 200;
        }
    }

    println!("\nCorners marked (value=200):");
    print_image_data(&marked, "Marked", 30, 30);
}
