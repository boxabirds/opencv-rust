/// Bit-level accuracy tests for Adaptive Threshold
mod test_utils;

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::{Scalar, ThresholdType};
use opencv_rust::imgproc::{adaptive_threshold, AdaptiveThresholdMethod};
use test_utils::*;

#[test]
fn test_adaptive_threshold_deterministic_mean() {
    let mut src = Mat::new(30, 30, 1, MatDepth::U8).unwrap();
    for i in 0..900 { src.data_mut()[i] = (i % 256) as u8; }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    adaptive_threshold(&src, &mut dst1, 255.0, AdaptiveThresholdMethod::Mean, ThresholdType::Binary, 11, 2.0).unwrap();
    adaptive_threshold(&src, &mut dst2, 255.0, AdaptiveThresholdMethod::Mean, ThresholdType::Binary, 11, 2.0).unwrap();

    assert_images_equal(&dst1, &dst2, "Adaptive threshold (mean) should be deterministic");
}

#[test]
fn test_adaptive_threshold_deterministic_gaussian() {
    let mut src = Mat::new(30, 30, 1, MatDepth::U8).unwrap();
    for i in 0..900 { src.data_mut()[i] = (i % 256) as u8; }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    adaptive_threshold(&src, &mut dst1, 255.0, AdaptiveThresholdMethod::Gaussian, ThresholdType::Binary, 11, 2.0).unwrap();
    adaptive_threshold(&src, &mut dst2, 255.0, AdaptiveThresholdMethod::Gaussian, ThresholdType::Binary, 11, 2.0).unwrap();

    assert_images_equal(&dst1, &dst2, "Adaptive threshold (gaussian) should be deterministic");
}

#[test]
fn test_adaptive_threshold_binary_output() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();
    for i in 0..400 { src.data_mut()[i] = ((i * 7) % 256) as u8; }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    adaptive_threshold(&src, &mut dst, 255.0, AdaptiveThresholdMethod::Mean, ThresholdType::Binary, 11, 2.0).unwrap();

    // Adaptive binary threshold should produce only 0 or 255
    for row in 0..20 {
        for col in 0..20 {
            let val = dst.at(row, col).unwrap()[0];
            assert!(val == 0 || val == 255,
                "Adaptive binary threshold at ({}, {}) should be 0 or 255, got {}", row, col, val);
        }
    }
}

#[test]
fn test_adaptive_threshold_binary_inv_output() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();
    for i in 0..400 { src.data_mut()[i] = ((i * 7) % 256) as u8; }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    adaptive_threshold(&src, &mut dst, 255.0, AdaptiveThresholdMethod::Mean, ThresholdType::BinaryInv, 11, 2.0).unwrap();

    // Binary inv should also produce only 0 or 255
    for row in 0..20 {
        for col in 0..20 {
            let val = dst.at(row, col).unwrap()[0];
            assert!(val == 0 || val == 255,
                "Adaptive binary inv at ({}, {}) should be 0 or 255, got {}", row, col, val);
        }
    }
}

#[test]
fn test_adaptive_threshold_block_sizes() {
    let mut src = Mat::new(40, 40, 1, MatDepth::U8).unwrap();
    for i in 0..1600 { src.data_mut()[i] = ((i * 5) % 256) as u8; }

    let mut dst3 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst11 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst21 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    adaptive_threshold(&src, &mut dst3, 255.0, AdaptiveThresholdMethod::Mean, ThresholdType::Binary, 3, 2.0).unwrap();
    adaptive_threshold(&src, &mut dst11, 255.0, AdaptiveThresholdMethod::Mean, ThresholdType::Binary, 11, 2.0).unwrap();
    adaptive_threshold(&src, &mut dst21, 255.0, AdaptiveThresholdMethod::Mean, ThresholdType::Binary, 21, 2.0).unwrap();

    assert_eq!(dst3.rows(), 40);
    assert_eq!(dst11.rows(), 40);
    assert_eq!(dst21.rows(), 40);
}

#[test]
fn test_adaptive_threshold_c_parameter() {
    let mut src = Mat::new(30, 30, 1, MatDepth::U8).unwrap();
    for i in 0..900 { src.data_mut()[i] = ((i * 7) % 256) as u8; }

    let mut dst_low = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst_high = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    adaptive_threshold(&src, &mut dst_low, 255.0, AdaptiveThresholdMethod::Mean, ThresholdType::Binary, 11, 2.0).unwrap();
    adaptive_threshold(&src, &mut dst_high, 255.0, AdaptiveThresholdMethod::Mean, ThresholdType::Binary, 11, 10.0).unwrap();

    // Different C values should produce different results
    let stats = compute_diff_stats(&dst_low, &dst_high);
    assert!(stats.diff_count > 0, "Different C values should produce different results");
}

#[test]
fn test_adaptive_threshold_gradient_image() {
    let mut src = Mat::new(30, 30, 1, MatDepth::U8).unwrap();

    // Create gradient
    for row in 0..30 {
        for col in 0..30 {
            src.at_mut(row, col).unwrap()[0] = ((row + col) * 4) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    adaptive_threshold(&src, &mut dst, 255.0, AdaptiveThresholdMethod::Mean, ThresholdType::Binary, 11, 2.0).unwrap();

    assert_eq!(dst.rows(), 30);
    assert_eq!(dst.cols(), 30);
}

#[test]
fn test_adaptive_threshold_method_comparison() {
    let mut src = Mat::new(30, 30, 1, MatDepth::U8).unwrap();
    for i in 0..900 { src.data_mut()[i] = ((i * 11) % 256) as u8; }

    let mut dst_mean = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst_gaussian = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    adaptive_threshold(&src, &mut dst_mean, 255.0, AdaptiveThresholdMethod::Mean, ThresholdType::Binary, 11, 2.0).unwrap();
    adaptive_threshold(&src, &mut dst_gaussian, 255.0, AdaptiveThresholdMethod::Gaussian, ThresholdType::Binary, 11, 2.0).unwrap();

    // Both methods should process without errors
    assert_eq!(dst_mean.rows(), 30);
    assert_eq!(dst_gaussian.rows(), 30);
}

#[test]
fn test_adaptive_threshold_boundary() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();
    for i in 0..100 { src.data_mut()[i] = ((i * 10) % 256) as u8; }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    adaptive_threshold(&src, &mut dst, 255.0, AdaptiveThresholdMethod::Mean, ThresholdType::Binary, 5, 2.0).unwrap();

    // Border pixels should be valid (0 or 255)
    for i in 0..10 {
        let top = dst.at(0, i).unwrap()[0];
        let bottom = dst.at(9, i).unwrap()[0];
        let left = dst.at(i, 0).unwrap()[0];
        let right = dst.at(i, 9).unwrap()[0];

        assert!(top == 0 || top == 255, "Top border should be binary");
        assert!(bottom == 0 || bottom == 255, "Bottom border should be binary");
        assert!(left == 0 || left == 255, "Left border should be binary");
        assert!(right == 0 || right == 255, "Right border should be binary");
    }
}

#[test]
#[ignore]
fn test_adaptive_threshold_visual_inspection() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();

    // Create pattern with varying local intensities
    for row in 0..20 {
        for col in 0..20 {
            let base = if col < 10 { 60 } else { 180 };
            let local_var = ((row * 7 + col * 11) % 30) as u8;
            src.at_mut(row, col).unwrap()[0] = base + local_var;
        }
    }

    println!("\nInput (varying local intensities):");
    print_image_data(&src, "Source", 20, 20);

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    adaptive_threshold(&src, &mut dst, 255.0, AdaptiveThresholdMethod::Mean, ThresholdType::Binary, 11, 2.0).unwrap();

    println!("\nAfter adaptive threshold (mean, block=11, C=2):");
    print_image_data(&dst, "Thresholded", 20, 20);
}
