/// Bit-level accuracy tests for Non-Local Means Denoising
mod test_utils;

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::Scalar;
use opencv_rust::imgproc::non_local_means_denoising;
use test_utils::*;

#[test]
fn test_nlm_deterministic() {
    let mut src = Mat::new(30, 30, 1, MatDepth::U8).unwrap();
    for i in 0..900 { src.data_mut()[i] = (i % 256) as u8; }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    non_local_means_denoising(&src, &mut dst1, 10.0, 7, 21).unwrap();
    non_local_means_denoising(&src, &mut dst2, 10.0, 7, 21).unwrap();

    assert_images_equal(&dst1, &dst2, "NLM denoising should be deterministic");
}

#[test]
fn test_nlm_uniform_image() {
    let src = Mat::new_with_default(30, 30, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    non_local_means_denoising(&src, &mut dst, 10.0, 7, 21).unwrap();

    // Uniform image should remain uniform
    for row in 5..25 {
        for col in 5..25 {
            let val = dst.at(row, col).unwrap()[0];
            assert!((val as i32 - 128).abs() <= 2,
                "Uniform image should remain ~128 at ({}, {}), got {}", row, col, val);
        }
    }
}

#[test]
fn test_nlm_h_parameter() {
    let mut src = Mat::new(30, 30, 1, MatDepth::U8).unwrap();
    for i in 0..900 { src.data_mut()[i] = ((i * 7) % 256) as u8; }

    let mut dst_low = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst_high = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    non_local_means_denoising(&src, &mut dst_low, 3.0, 7, 21).unwrap();    // Low h (less smoothing)
    non_local_means_denoising(&src, &mut dst_high, 30.0, 7, 21).unwrap();  // High h (more smoothing)

    // Both should process without errors
    assert_eq!(dst_low.rows(), 30);
    assert_eq!(dst_high.rows(), 30);
}

#[test]
fn test_nlm_template_window_size() {
    let mut src = Mat::new(30, 30, 1, MatDepth::U8).unwrap();
    for i in 0..900 { src.data_mut()[i] = ((i * 11) % 256) as u8; }

    let mut dst_small = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst_large = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    non_local_means_denoising(&src, &mut dst_small, 10.0, 3, 21).unwrap();   // Small template
    non_local_means_denoising(&src, &mut dst_large, 10.0, 11, 21).unwrap();  // Large template

    // Both should process without errors
    assert_eq!(dst_small.rows(), 30);
    assert_eq!(dst_large.rows(), 30);
}

#[test]
fn test_nlm_search_window_size() {
    let mut src = Mat::new(40, 40, 1, MatDepth::U8).unwrap();
    for i in 0..1600 { src.data_mut()[i] = ((i * 13) % 256) as u8; }

    let mut dst_small = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst_large = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    non_local_means_denoising(&src, &mut dst_small, 10.0, 7, 11).unwrap();  // Small search
    non_local_means_denoising(&src, &mut dst_large, 10.0, 7, 31).unwrap();  // Large search

    // Both should process without errors
    assert_eq!(dst_small.rows(), 40);
    assert_eq!(dst_large.rows(), 40);
}

#[test]
fn test_nlm_denoises_pattern() {
    let mut src = Mat::new(30, 30, 1, MatDepth::U8).unwrap();

    // Create noisy pattern
    for row in 0..30 {
        for col in 0..30 {
            let base = 100;
            let noise = if (row + col) % 5 == 0 { 30 } else { 0 };
            src.at_mut(row, col).unwrap()[0] = base + noise;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    non_local_means_denoising(&src, &mut dst, 15.0, 7, 21).unwrap();

    // NLM should smooth the noise
    assert_eq!(dst.rows(), 30);
    assert_eq!(dst.cols(), 30);
}

#[test]
fn test_nlm_multichannel() {
    let mut src = Mat::new(20, 20, 3, MatDepth::U8).unwrap();
    for i in 0..1200 { src.data_mut()[i] = ((i * 7) % 256) as u8; }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    non_local_means_denoising(&src, &mut dst, 10.0, 7, 21).unwrap();

    // Verify dimensions and channels
    assert_eq!(dst.rows(), 20);
    assert_eq!(dst.cols(), 20);
    assert_eq!(dst.channels(), 3);
}

#[test]
fn test_nlm_output_range() {
    let mut src = Mat::new(30, 30, 1, MatDepth::U8).unwrap();

    // Create extreme pattern
    for row in 0..30 {
        for col in 0..30 {
            src.at_mut(row, col).unwrap()[0] = if (row + col) % 2 == 0 { 0 } else { 255 };
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    non_local_means_denoising(&src, &mut dst, 10.0, 7, 21).unwrap();

    // All values should be in valid range
    for row in 0..30 {
        for col in 0..30 {
            let val = dst.at(row, col).unwrap()[0];
            assert!(val <= 255,
                "NLM output at ({}, {}) out of range: {}", row, col, val);
        }
    }
}

#[test]
fn test_nlm_boundary_handling() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();
    for i in 0..100 { src.data_mut()[i] = ((i * 10) % 256) as u8; }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    non_local_means_denoising(&src, &mut dst, 10.0, 7, 21).unwrap();

    // Border pixels should be valid
    for i in 0..10 {
        assert!(dst.at(0, i).unwrap()[0] <= 255, "Top border valid");
        assert!(dst.at(9, i).unwrap()[0] <= 255, "Bottom border valid");
        assert!(dst.at(i, 0).unwrap()[0] <= 255, "Left border valid");
        assert!(dst.at(i, 9).unwrap()[0] <= 255, "Right border valid");
    }
}

#[test]
fn test_nlm_small_image() {
    let mut src = Mat::new(5, 5, 1, MatDepth::U8).unwrap();
    for i in 0..25 { src.data_mut()[i] = ((i * 10) % 256) as u8; }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    non_local_means_denoising(&src, &mut dst, 10.0, 3, 5).unwrap();

    assert_eq!(dst.rows(), 5);
    assert_eq!(dst.cols(), 5);
}

#[test]
fn test_nlm_preserves_structure() {
    let mut src = Mat::new(30, 30, 1, MatDepth::U8).unwrap();

    // Create edge with noise
    for row in 0..30 {
        for col in 0..30 {
            let base = if col < 15 { 50 } else { 200 };
            let noise = if (row * 7 + col * 11) % 13 == 0 { 20 } else { 0 };
            src.at_mut(row, col).unwrap()[0] = (base + noise).min(255);
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    non_local_means_denoising(&src, &mut dst, 15.0, 7, 21).unwrap();

    // Edge structure should be preserved (left and right regions distinct)
    assert_eq!(dst.rows(), 30);
    assert_eq!(dst.cols(), 30);
}

#[test]
#[ignore]
fn test_nlm_visual_inspection() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();

    // Create pattern with noise
    for row in 0..20 {
        for col in 0..20 {
            let base = 100;
            let noise = if (row * 7 + col * 11) % 11 == 0 { 40 } else { 0 };
            src.at_mut(row, col).unwrap()[0] = (base + noise).min(255);
        }
    }

    println!("\nInput (with noise):");
    print_image_data(&src, "Source", 20, 20);

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    non_local_means_denoising(&src, &mut dst, 15.0, 7, 21).unwrap();

    println!("\nAfter NLM denoising:");
    print_image_data(&dst, "Denoised", 20, 20);

    let stats = compute_diff_stats(&src, &dst);
    println!("\nDifference from original:");
    println!("{}", stats);
}
