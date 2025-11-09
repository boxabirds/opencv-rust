/// Bit-level accuracy tests for Warp Affine
mod test_utils;

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::{Scalar, Size, Point2f};
use opencv_rust::imgproc::{warp_affine, get_rotation_matrix_2d, get_affine_transform};
use test_utils::*;

#[test]
fn test_warp_affine_identity_deterministic() {
    let mut src = Mat::new(20, 30, 1, MatDepth::U8).unwrap();
    for i in 0..600 { src.data_mut()[i] = (i % 256) as u8; }

    // Identity transformation
    let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    warp_affine(&src, &mut dst1, &m, Size::new(30, 20)).unwrap();
    warp_affine(&src, &mut dst2, &m, Size::new(30, 20)).unwrap();

    assert_images_equal(&dst1, &dst2, "Warp affine should be deterministic");
}

#[test]
fn test_warp_affine_identity_preserves_image() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();
    for row in 0..20 {
        for col in 0..20 {
            src.at_mut(row, col).unwrap()[0] = (row * 10 + col) as u8;
        }
    }

    // Identity transformation
    let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    warp_affine(&src, &mut dst, &m, Size::new(20, 20)).unwrap();

    assert_images_equal(&src, &dst, "Identity transform should preserve image");
}

#[test]
fn test_warp_affine_translation() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();

    // Create unique pattern at position (5,5)
    src.at_mut(5, 5).unwrap()[0] = 255;
    src.at_mut(5, 6).unwrap()[0] = 200;
    src.at_mut(6, 5).unwrap()[0] = 150;

    // Translate by (5, 5) - affine uses backward mapping so positive translation brings content from (5,5) to (0,0)
    let m = [[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]];
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    warp_affine(&src, &mut dst, &m, Size::new(20, 20)).unwrap();

    // Warp affine with positive translation in matrix pulls pixels from higher coordinates
    // So dst(0,0) = src(5,5)
    assert_eq!(dst.at(0, 0).unwrap()[0], 255, "Translated pixel should match");
    assert_eq!(dst.at(0, 1).unwrap()[0], 200, "Translated pixel should match");
    assert_eq!(dst.at(1, 0).unwrap()[0], 150, "Translated pixel should match");
}

#[test]
fn test_warp_affine_scaling() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();
    for i in 0..100 { src.data_mut()[i] = ((i * 10) % 256) as u8; }

    // Scale by 2x
    let m = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    warp_affine(&src, &mut dst, &m, Size::new(20, 20)).unwrap();

    assert_eq!(dst.rows(), 20, "Destination should be scaled");
    assert_eq!(dst.cols(), 20, "Destination should be scaled");
}

#[test]
fn test_warp_affine_rotation_matrix() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();
    for i in 0..400 { src.data_mut()[i] = (i % 256) as u8; }

    // Get rotation matrix (rotate around center)
    let center = Point2f { x: 10.0, y: 10.0 };
    let m = get_rotation_matrix_2d(center, 90.0, 1.0);

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    warp_affine(&src, &mut dst, &m, Size::new(20, 20)).unwrap();

    assert_eq!(dst.rows(), 20);
    assert_eq!(dst.cols(), 20);
}

#[test]
fn test_warp_affine_get_affine_transform() {
    let mut src = Mat::new(20, 20, 1, MatDepth::U8).unwrap();
    for i in 0..400 { src.data_mut()[i] = (i % 256) as u8; }

    // Define source and destination points for affine transform
    let src_pts = [
        Point2f { x: 0.0, y: 0.0 },
        Point2f { x: 19.0, y: 0.0 },
        Point2f { x: 0.0, y: 19.0 },
    ];
    let dst_pts = [
        Point2f { x: 0.0, y: 0.0 },
        Point2f { x: 19.0, y: 0.0 },
        Point2f { x: 0.0, y: 19.0 },
    ];

    let m = get_affine_transform(&src_pts, &dst_pts);
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    warp_affine(&src, &mut dst, &m, Size::new(20, 20)).unwrap();

    // Identity points should produce identity-like result
    assert_eq!(dst.rows(), 20);
    assert_eq!(dst.cols(), 20);
}

#[test]
fn test_warp_affine_output_size() {
    let src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();
    let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    warp_affine(&src, &mut dst, &m, Size::new(30, 25)).unwrap();

    assert_eq!(dst.rows(), 25, "Output should match requested size");
    assert_eq!(dst.cols(), 30, "Output should match requested size");
}

#[test]
fn test_warp_affine_multichannel() {
    let mut src = Mat::new(10, 10, 3, MatDepth::U8).unwrap();
    for i in 0..300 { src.data_mut()[i] = (i % 256) as u8; }

    let m = [[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]];
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    warp_affine(&src, &mut dst, &m, Size::new(15, 15)).unwrap();

    assert_eq!(dst.channels(), 3, "Channels should be preserved");
}

#[test]
fn test_warp_affine_boundary_handling() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();
    for i in 0..100 { src.data_mut()[i] = 100; }

    // Translate beyond boundaries
    let m = [[1.0, 0.0, 20.0], [0.0, 1.0, 20.0]];
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    warp_affine(&src, &mut dst, &m, Size::new(10, 10)).unwrap();

    // Pixels outside source should be 0 (default background)
    assert_eq!(dst.at(0, 0).unwrap()[0], 0, "Out-of-bounds should be 0");
}

#[test]
fn test_warp_affine_uniform_image() {
    let src = Mat::new_with_default(20, 20, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
    let m = [[1.0, 0.0, 5.0], [0.0, 1.0, 5.0]];
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    warp_affine(&src, &mut dst, &m, Size::new(20, 20)).unwrap();

    // Translated uniform region should still be uniform
    for row in 5..15 {
        for col in 5..15 {
            assert_eq!(dst.at(row, col).unwrap()[0], 128,
                "Uniform region should remain uniform at ({}, {})", row, col);
        }
    }
}

#[test]
fn test_warp_affine_small_image() {
    let mut src = Mat::new(5, 5, 1, MatDepth::U8).unwrap();
    for i in 0..25 { src.data_mut()[i] = (i * 10) as u8; }

    let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    warp_affine(&src, &mut dst, &m, Size::new(5, 5)).unwrap();

    assert_eq!(dst.rows(), 5);
    assert_eq!(dst.cols(), 5);
}

#[test]
#[ignore]
fn test_warp_affine_visual_inspection() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();
    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 20 + col * 10) as u8;
        }
    }

    println!("\nOriginal:");
    print_image_data(&src, "Source", 10, 10);

    // Translation
    let m_trans = [[1.0, 0.0, 3.0], [0.0, 1.0, 3.0]];
    let mut dst_trans = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    warp_affine(&src, &mut dst_trans, &m_trans, Size::new(10, 10)).unwrap();
    println!("\nTranslated by (3,3):");
    print_image_data(&dst_trans, "Translated", 10, 10);

    // Rotation
    let center = Point2f { x: 5.0, y: 5.0 };
    let m_rot = get_rotation_matrix_2d(center, 45.0, 1.0);
    let mut dst_rot = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    warp_affine(&src, &mut dst_rot, &m_rot, Size::new(10, 10)).unwrap();
    println!("\nRotated 45°:");
    print_image_data(&dst_rot, "Rotated", 10, 10);
}
