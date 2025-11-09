/// Bit-level accuracy tests for Flip
/// These tests verify that optimizations don't change results
mod test_utils;

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::Scalar;
use opencv_rust::imgproc::flip;
use test_utils::*;

/// Test flip vertical is deterministic
#[test]
fn test_flip_vertical_deterministic() {
    let mut src = Mat::new(20, 30, 1, MatDepth::U8).unwrap();

    for row in 0..20 {
        for col in 0..30 {
            src.at_mut(row, col).unwrap()[0] = ((row * 10 + col) % 256) as u8;
        }
    }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    flip(&src, &mut dst1, 0).unwrap(); // Vertical flip
    flip(&src, &mut dst2, 0).unwrap();

    assert_images_equal(&dst1, &dst2, "Vertical flip should be deterministic");
}

/// Test flip horizontal is deterministic
#[test]
fn test_flip_horizontal_deterministic() {
    let mut src = Mat::new(20, 30, 1, MatDepth::U8).unwrap();

    for row in 0..20 {
        for col in 0..30 {
            src.at_mut(row, col).unwrap()[0] = ((row * 10 + col) % 256) as u8;
        }
    }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    flip(&src, &mut dst1, 1).unwrap(); // Horizontal flip
    flip(&src, &mut dst2, 1).unwrap();

    assert_images_equal(&dst1, &dst2, "Horizontal flip should be deterministic");
}

/// Test flip both is deterministic
#[test]
fn test_flip_both_deterministic() {
    let mut src = Mat::new(20, 30, 1, MatDepth::U8).unwrap();

    for row in 0..20 {
        for col in 0..30 {
            src.at_mut(row, col).unwrap()[0] = ((row * 10 + col) % 256) as u8;
        }
    }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    flip(&src, &mut dst1, -1).unwrap(); // Both axes flip
    flip(&src, &mut dst2, -1).unwrap();

    assert_images_equal(&dst1, &dst2, "Both-axes flip should be deterministic");
}

/// Test vertical flip correctness
#[test]
fn test_flip_vertical_correctness() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    // Create pattern where each row has a unique value
    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 10) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    flip(&src, &mut dst, 0).unwrap();

    // Verify first row of dst matches last row of src
    for col in 0..10 {
        assert_eq!(
            dst.at(0, col).unwrap()[0],
            src.at(9, col).unwrap()[0],
            "Vertical flip: first row should match source last row at col {}",
            col
        );
    }

    // Verify last row of dst matches first row of src
    for col in 0..10 {
        assert_eq!(
            dst.at(9, col).unwrap()[0],
            src.at(0, col).unwrap()[0],
            "Vertical flip: last row should match source first row at col {}",
            col
        );
    }
}

/// Test horizontal flip correctness
#[test]
fn test_flip_horizontal_correctness() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    // Create pattern where each column has a unique value
    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = col as u8 * 10;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    flip(&src, &mut dst, 1).unwrap();

    // Verify first column of dst matches last column of src
    for row in 0..10 {
        assert_eq!(
            dst.at(row, 0).unwrap()[0],
            src.at(row, 9).unwrap()[0],
            "Horizontal flip: first col should match source last col at row {}",
            row
        );
    }

    // Verify last column of dst matches first column of src
    for row in 0..10 {
        assert_eq!(
            dst.at(row, 9).unwrap()[0],
            src.at(row, 0).unwrap()[0],
            "Horizontal flip: last col should match source first col at row {}",
            row
        );
    }
}

/// Test both-axes flip correctness
#[test]
fn test_flip_both_correctness() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    // Create unique pattern
    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 10 + col) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    flip(&src, &mut dst, -1).unwrap();

    // Top-left of dst should match bottom-right of src
    assert_eq!(
        dst.at(0, 0).unwrap()[0],
        src.at(9, 9).unwrap()[0],
        "Both flip: (0,0) should match source (9,9)"
    );

    // Bottom-right of dst should match top-left of src
    assert_eq!(
        dst.at(9, 9).unwrap()[0],
        src.at(0, 0).unwrap()[0],
        "Both flip: (9,9) should match source (0,0)"
    );
}

/// Test double flip returns to original
#[test]
fn test_flip_double_flip() {
    let mut src = Mat::new(15, 15, 1, MatDepth::U8).unwrap();

    for row in 0..15 {
        for col in 0..15 {
            src.at_mut(row, col).unwrap()[0] = ((row * 7 + col * 13) % 256) as u8;
        }
    }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    // Flip twice should return to original
    flip(&src, &mut dst1, 0).unwrap();
    flip(&dst1, &mut dst2, 0).unwrap();

    assert_images_equal(&src, &dst2, "Double vertical flip should return to original");
}

/// Test flip preserves dimensions
#[test]
fn test_flip_preserves_dimensions() {
    let mut src = Mat::new(25, 30, 3, MatDepth::U8).unwrap();

    for i in 0..(25 * 30 * 3) {
        src.data_mut()[i] = (i % 256) as u8;
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    flip(&src, &mut dst, 0).unwrap();

    assert_eq!(dst.rows(), 25, "Flip should preserve rows");
    assert_eq!(dst.cols(), 30, "Flip should preserve cols");
    assert_eq!(dst.channels(), 3, "Flip should preserve channels");
}

/// Test flip on multi-channel image
#[test]
fn test_flip_multichannel() {
    let mut src = Mat::new(10, 10, 3, MatDepth::U8).unwrap();

    // Create RGB pattern
    for row in 0..10 {
        for col in 0..10 {
            let pixel = src.at_mut(row, col).unwrap();
            pixel[0] = row as u8 * 10;        // Red by row
            pixel[1] = col as u8 * 10;        // Green by col
            pixel[2] = 128;                    // Blue constant
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    flip(&src, &mut dst, 1).unwrap(); // Horizontal flip

    // Verify channels are flipped correctly
    assert_eq!(dst.at(0, 0).unwrap()[0], src.at(0, 9).unwrap()[0], "Red channel flipped");
    assert_eq!(dst.at(0, 0).unwrap()[1], src.at(0, 9).unwrap()[1], "Green channel flipped");
    assert_eq!(dst.at(0, 0).unwrap()[2], 128, "Blue channel preserved");
}

/// Test flip on uniform image
#[test]
fn test_flip_uniform_image() {
    let src = Mat::new_with_default(20, 20, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    flip(&src, &mut dst, 0).unwrap();

    // Uniform image should remain uniform after flip
    for row in 0..20 {
        for col in 0..20 {
            assert_eq!(dst.at(row, col).unwrap()[0], 128,
                "Uniform image should remain uniform at ({}, {})", row, col);
        }
    }
}

/// Test flip on small image
#[test]
fn test_flip_small_image() {
    let mut src = Mat::new(3, 3, 1, MatDepth::U8).unwrap();

    for i in 0..9 {
        src.data_mut()[i] = (i * 10) as u8;
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    flip(&src, &mut dst, -1).unwrap();

    assert_eq!(dst.rows(), 3);
    assert_eq!(dst.cols(), 3);
}

/// Test flip symmetry (vertical + horizontal = both)
#[test]
fn test_flip_symmetry() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 10 + col) as u8;
        }
    }

    // Flip vertical then horizontal
    let mut temp = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut vh_flip = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    flip(&src, &mut temp, 0).unwrap();
    flip(&temp, &mut vh_flip, 1).unwrap();

    // Flip both at once
    let mut both_flip = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    flip(&src, &mut both_flip, -1).unwrap();

    // Should produce same result
    assert_images_equal(&vh_flip, &both_flip,
        "Vertical+horizontal should equal both-axes flip");
}

/// Visual inspection test (ignored by default)
#[test]
#[ignore]
fn test_flip_visual_inspection() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    // Create gradient pattern
    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 20 + col * 5) as u8;
        }
    }

    println!("\nOriginal:");
    print_image_data(&src, "Source", 10, 10);

    let mut dst_v = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    flip(&src, &mut dst_v, 0).unwrap();
    println!("\nVertical flip:");
    print_image_data(&dst_v, "Vertical", 10, 10);

    let mut dst_h = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    flip(&src, &mut dst_h, 1).unwrap();
    println!("\nHorizontal flip:");
    print_image_data(&dst_h, "Horizontal", 10, 10);

    let mut dst_b = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    flip(&src, &mut dst_b, -1).unwrap();
    println!("\nBoth axes flip:");
    print_image_data(&dst_b, "Both", 10, 10);
}
