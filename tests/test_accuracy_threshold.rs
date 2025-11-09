/// Bit-level accuracy tests for Threshold operations
/// These tests verify that optimizations don't change results
mod test_utils;

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::Scalar;
use opencv_rust::imgproc::threshold;
use opencv_rust::core::types::ThresholdType;
use test_utils::*;

/// Test binary threshold is deterministic
#[test]
fn test_threshold_binary_deterministic() {
    let mut src = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Create gradient
    for row in 0..50 {
        for col in 0..50 {
            src.at_mut(row, col).unwrap()[0] = ((row * 5 + col) % 256) as u8;
        }
    }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    threshold(&src, &mut dst1, 128.0, 255.0, ThresholdType::Binary).unwrap();
    threshold(&src, &mut dst2, 128.0, 255.0, ThresholdType::Binary).unwrap();

    // Results should be bit-exact identical
    assert_images_equal(&dst1, &dst2, "Binary threshold should be deterministic");
}

/// Test binary threshold correctness
#[test]
fn test_threshold_binary_correctness() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    // Create known values
    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 10 + col * 5) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    threshold(&src, &mut dst, 100.0, 200.0, ThresholdType::Binary).unwrap();

    // Verify each pixel
    for row in 0..10 {
        for col in 0..10 {
            let src_val = src.at(row, col).unwrap()[0];
            let dst_val = dst.at(row, col).unwrap()[0];
            let expected = if src_val > 100 { 200 } else { 0 };

            assert_eq!(dst_val, expected,
                "Binary threshold at ({}, {}): src={}, expected={}, got={}",
                row, col, src_val, expected, dst_val);
        }
    }
}

/// Test binary inverted threshold correctness
#[test]
fn test_threshold_binary_inv_correctness() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 20) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    threshold(&src, &mut dst, 100.0, 255.0, ThresholdType::BinaryInv).unwrap();

    // Verify inversion
    for row in 0..10 {
        for col in 0..10 {
            let src_val = src.at(row, col).unwrap()[0];
            let dst_val = dst.at(row, col).unwrap()[0];
            let expected = if src_val > 100 { 0 } else { 255 };

            assert_eq!(dst_val, expected,
                "BinaryInv threshold at ({}, {}): src={}, expected={}, got={}",
                row, col, src_val, expected, dst_val);
        }
    }
}

/// Test truncate threshold correctness
#[test]
fn test_threshold_trunc_correctness() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 25) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    threshold(&src, &mut dst, 150.0, 0.0, ThresholdType::Trunc).unwrap();

    // Verify truncation
    for row in 0..10 {
        for col in 0..10 {
            let src_val = src.at(row, col).unwrap()[0];
            let dst_val = dst.at(row, col).unwrap()[0];
            let expected = if src_val > 150 { 150 } else { src_val };

            assert_eq!(dst_val, expected,
                "Trunc threshold at ({}, {}): src={}, expected={}, got={}",
                row, col, src_val, expected, dst_val);
        }
    }
}

/// Test to-zero threshold correctness
#[test]
fn test_threshold_to_zero_correctness() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 15 + col * 10) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    threshold(&src, &mut dst, 80.0, 0.0, ThresholdType::ToZero).unwrap();

    // Verify to-zero behavior
    for row in 0..10 {
        for col in 0..10 {
            let src_val = src.at(row, col).unwrap()[0];
            let dst_val = dst.at(row, col).unwrap()[0];
            let expected = if src_val > 80 { src_val } else { 0 };

            assert_eq!(dst_val, expected,
                "ToZero threshold at ({}, {}): src={}, expected={}, got={}",
                row, col, src_val, expected, dst_val);
        }
    }
}

/// Test to-zero inverted threshold correctness
#[test]
fn test_threshold_to_zero_inv_correctness() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 12 + col * 8) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    threshold(&src, &mut dst, 100.0, 0.0, ThresholdType::ToZeroInv).unwrap();

    // Verify to-zero inverted behavior
    for row in 0..10 {
        for col in 0..10 {
            let src_val = src.at(row, col).unwrap()[0];
            let dst_val = dst.at(row, col).unwrap()[0];
            let expected = if src_val > 100 { 0 } else { src_val };

            assert_eq!(dst_val, expected,
                "ToZeroInv threshold at ({}, {}): src={}, expected={}, got={}",
                row, col, src_val, expected, dst_val);
        }
    }
}

/// Test threshold on uniform image
#[test]
fn test_threshold_uniform_below() {
    let src = Mat::new_with_default(20, 20, 1, MatDepth::U8, Scalar::all(50.0)).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    threshold(&src, &mut dst, 100.0, 255.0, ThresholdType::Binary).unwrap();

    // All pixels should be 0 (below threshold)
    for row in 0..20 {
        for col in 0..20 {
            assert_eq!(dst.at(row, col).unwrap()[0], 0,
                "Pixel ({}, {}) should be 0", row, col);
        }
    }
}

/// Test threshold on uniform image above threshold
#[test]
fn test_threshold_uniform_above() {
    let src = Mat::new_with_default(20, 20, 1, MatDepth::U8, Scalar::all(200.0)).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    threshold(&src, &mut dst, 100.0, 255.0, ThresholdType::Binary).unwrap();

    // All pixels should be 255 (above threshold)
    for row in 0..20 {
        for col in 0..20 {
            assert_eq!(dst.at(row, col).unwrap()[0], 255,
                "Pixel ({}, {}) should be 255", row, col);
        }
    }
}

/// Test threshold on multi-channel image
#[test]
fn test_threshold_multichannel_independence() {
    let mut src = Mat::new(20, 20, 3, MatDepth::U8).unwrap();

    // Each channel different pattern
    for row in 0..20 {
        for col in 0..20 {
            let pixel = src.at_mut(row, col).unwrap();
            pixel[0] = (row * 10) as u8;     // 0-190
            pixel[1] = 200;                   // Constant high
            pixel[2] = 50;                    // Constant low
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    threshold(&src, &mut dst, 100.0, 255.0, ThresholdType::Binary).unwrap();

    // Verify independent channel processing
    for row in 0..20 {
        for col in 0..20 {
            let pixel = dst.at(row, col).unwrap();
            let expected_r = if src.at(row, col).unwrap()[0] > 100 { 255 } else { 0 };

            assert_eq!(pixel[0], expected_r, "Red channel at ({}, {})", row, col);
            assert_eq!(pixel[1], 255, "Green channel should always be 255 at ({}, {})", row, col);
            assert_eq!(pixel[2], 0, "Blue channel should always be 0 at ({}, {})", row, col);
        }
    }
}

/// Test threshold boundary conditions
#[test]
fn test_threshold_boundary_exact() {
    let mut src = Mat::new(5, 5, 1, MatDepth::U8).unwrap();

    // Values exactly at and around threshold
    for i in 0..25 {
        src.data_mut()[i] = 100;
    }
    src.at_mut(2, 2).unwrap()[0] = 99;  // Just below
    src.at_mut(2, 3).unwrap()[0] = 101; // Just above

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    threshold(&src, &mut dst, 100.0, 255.0, ThresholdType::Binary).unwrap();

    // Value exactly at threshold should NOT trigger (> not >=)
    assert_eq!(dst.at(0, 0).unwrap()[0], 0, "Value at threshold (100) should be 0");
    assert_eq!(dst.at(2, 2).unwrap()[0], 0, "Value below threshold (99) should be 0");
    assert_eq!(dst.at(2, 3).unwrap()[0], 255, "Value above threshold (101) should be 255");
}

/// Test threshold with extreme values
#[test]
fn test_threshold_extreme_values() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    // Mix of 0 and 255
    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = if (row + col) % 2 == 0 { 0 } else { 255 };
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    threshold(&src, &mut dst, 128.0, 200.0, ThresholdType::Binary).unwrap();

    // Verify extreme value handling
    for row in 0..10 {
        for col in 0..10 {
            let src_val = src.at(row, col).unwrap()[0];
            let dst_val = dst.at(row, col).unwrap()[0];

            if src_val == 0 {
                assert_eq!(dst_val, 0, "0 should remain 0");
            } else {
                assert_eq!(dst_val, 200, "255 should become 200");
            }
        }
    }
}

/// Test threshold preserves dimensions
#[test]
fn test_threshold_dimensions() {
    let src = Mat::new_with_default(73, 127, 3, MatDepth::U8, Scalar::all(100.0)).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    threshold(&src, &mut dst, 50.0, 255.0, ThresholdType::Binary).unwrap();

    assert_eq!(dst.rows(), 73);
    assert_eq!(dst.cols(), 127);
    assert_eq!(dst.channels(), 3);
}

/// Test all threshold types on same input
#[test]
fn test_threshold_all_types_consistency() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    for i in 0..100 {
        src.data_mut()[i] = (i * 2) as u8;
    }

    let mut dst_binary = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst_binary_inv = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst_trunc = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst_tozero = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst_tozero_inv = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    threshold(&src, &mut dst_binary, 100.0, 255.0, ThresholdType::Binary).unwrap();
    threshold(&src, &mut dst_binary_inv, 100.0, 255.0, ThresholdType::BinaryInv).unwrap();
    threshold(&src, &mut dst_trunc, 100.0, 0.0, ThresholdType::Trunc).unwrap();
    threshold(&src, &mut dst_tozero, 100.0, 0.0, ThresholdType::ToZero).unwrap();
    threshold(&src, &mut dst_tozero_inv, 100.0, 0.0, ThresholdType::ToZeroInv).unwrap();

    // Verify logical relationships between different types
    for i in 0..100 {
        let src_val = src.data()[i];

        if src_val > 100 {
            assert_eq!(dst_binary.data()[i], 255, "Binary above threshold");
            assert_eq!(dst_binary_inv.data()[i], 0, "BinaryInv above threshold");
            assert_eq!(dst_trunc.data()[i], 100, "Trunc above threshold");
            assert_eq!(dst_tozero.data()[i], src_val, "ToZero above threshold");
            assert_eq!(dst_tozero_inv.data()[i], 0, "ToZeroInv above threshold");
        } else {
            assert_eq!(dst_binary.data()[i], 0, "Binary below threshold");
            assert_eq!(dst_binary_inv.data()[i], 255, "BinaryInv below threshold");
            assert_eq!(dst_trunc.data()[i], src_val, "Trunc below threshold");
            assert_eq!(dst_tozero.data()[i], 0, "ToZero below threshold");
            assert_eq!(dst_tozero_inv.data()[i], src_val, "ToZeroInv below threshold");
        }
    }
}

/// Visual inspection test (ignored by default)
#[test]
#[ignore]
fn test_threshold_visual_inspection() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    // Create gradient
    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 25) as u8;
        }
    }

    println!("\nOriginal (gradient):");
    print_image_data(&src, "Source", 10, 10);

    let mut dst_binary = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    threshold(&src, &mut dst_binary, 128.0, 255.0, ThresholdType::Binary).unwrap();

    println!("\nBinary threshold (thresh=128, maxval=255):");
    print_image_data(&dst_binary, "Binary", 10, 10);

    let mut dst_trunc = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    threshold(&src, &mut dst_trunc, 128.0, 0.0, ThresholdType::Trunc).unwrap();

    println!("\nTrunc threshold (thresh=128):");
    print_image_data(&dst_trunc, "Trunc", 10, 10);

    let mut dst_tozero = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    threshold(&src, &mut dst_tozero, 128.0, 0.0, ThresholdType::ToZero).unwrap();

    println!("\nToZero threshold (thresh=128):");
    print_image_data(&dst_tozero, "ToZero", 10, 10);
}
