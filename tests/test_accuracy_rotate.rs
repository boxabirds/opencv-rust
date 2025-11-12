#![allow(unused_comparisons)]
/// Bit-level accuracy tests for Rotate
mod test_utils;

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::Scalar;
use opencv_rust::imgproc::{rotate, RotateCode};
use test_utils::*;

#[test]
fn test_rotate_90cw_deterministic() {
    let mut src = Mat::new(20, 30, 1, MatDepth::U8).unwrap();
    for i in 0..600 { src.data_mut()[i] = (i % 256) as u8; }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    rotate(&src, &mut dst1, RotateCode::Rotate90Clockwise).unwrap();
    rotate(&src, &mut dst2, RotateCode::Rotate90Clockwise).unwrap();

    assert_images_equal(&dst1, &dst2, "90CW rotate should be deterministic");
}

#[test]
fn test_rotate_180_deterministic() {
    let mut src = Mat::new(20, 30, 1, MatDepth::U8).unwrap();
    for i in 0..600 { src.data_mut()[i] = (i % 256) as u8; }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    rotate(&src, &mut dst1, RotateCode::Rotate180).unwrap();
    rotate(&src, &mut dst2, RotateCode::Rotate180).unwrap();

    assert_images_equal(&dst1, &dst2, "180 rotate should be deterministic");
}

#[test]
fn test_rotate_90ccw_deterministic() {
    let mut src = Mat::new(20, 30, 1, MatDepth::U8).unwrap();
    for i in 0..600 { src.data_mut()[i] = (i % 256) as u8; }

    let mut dst1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    rotate(&src, &mut dst1, RotateCode::Rotate90CounterClockwise).unwrap();
    rotate(&src, &mut dst2, RotateCode::Rotate90CounterClockwise).unwrap();

    assert_images_equal(&dst1, &dst2, "90CCW rotate should be deterministic");
}

#[test]
fn test_rotate_90cw_dimensions() {
    let src = Mat::new(20, 30, 1, MatDepth::U8).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    rotate(&src, &mut dst, RotateCode::Rotate90Clockwise).unwrap();

    assert_eq!(dst.rows(), 30, "90CW: rows should be src cols");
    assert_eq!(dst.cols(), 20, "90CW: cols should be src rows");
}

#[test]
fn test_rotate_180_dimensions() {
    let src = Mat::new(20, 30, 1, MatDepth::U8).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    rotate(&src, &mut dst, RotateCode::Rotate180).unwrap();

    assert_eq!(dst.rows(), 20, "180: rows should remain same");
    assert_eq!(dst.cols(), 30, "180: cols should remain same");
}

#[test]
fn test_rotate_4x90cw_identity() {
    let mut src = Mat::new(15, 15, 1, MatDepth::U8).unwrap();
    for i in 0..225 { src.data_mut()[i] = (i % 256) as u8; }

    let mut temp1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut temp2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut temp3 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    rotate(&src, &mut temp1, RotateCode::Rotate90Clockwise).unwrap();
    rotate(&temp1, &mut temp2, RotateCode::Rotate90Clockwise).unwrap();
    rotate(&temp2, &mut temp3, RotateCode::Rotate90Clockwise).unwrap();
    rotate(&temp3, &mut dst, RotateCode::Rotate90Clockwise).unwrap();

    assert_images_equal(&src, &dst, "4x 90CW should return to original");
}

#[test]
fn test_rotate_2x180_identity() {
    let mut src = Mat::new(15, 20, 1, MatDepth::U8).unwrap();
    for i in 0..300 { src.data_mut()[i] = (i % 256) as u8; }

    let mut temp = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    rotate(&src, &mut temp, RotateCode::Rotate180).unwrap();
    rotate(&temp, &mut dst, RotateCode::Rotate180).unwrap();

    assert_images_equal(&src, &dst, "2x 180 should return to original");
}

#[test]
fn test_rotate_90cw_90ccw_identity() {
    let mut src = Mat::new(15, 20, 1, MatDepth::U8).unwrap();
    for i in 0..300 { src.data_mut()[i] = (i % 256) as u8; }

    let mut temp = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    rotate(&src, &mut temp, RotateCode::Rotate90Clockwise).unwrap();
    rotate(&temp, &mut dst, RotateCode::Rotate90CounterClockwise).unwrap();

    assert_images_equal(&src, &dst, "90CW + 90CCW should return to original");
}

#[test]
fn test_rotate_multichannel() {
    let mut src = Mat::new(10, 10, 3, MatDepth::U8).unwrap();
    for i in 0..300 { src.data_mut()[i] = (i % 256) as u8; }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    rotate(&src, &mut dst, RotateCode::Rotate90Clockwise).unwrap();

    assert_eq!(dst.channels(), 3, "Channels should be preserved");
}

#[test]
fn test_rotate_uniform_image() {
    let src = Mat::new_with_default(20, 20, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    rotate(&src, &mut dst, RotateCode::Rotate180).unwrap();

    assert_images_equal(&src, &dst, "Uniform image should remain unchanged after 180 rotation");
}

#[test]
fn test_rotate_90cw_correctness() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();
    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 10 + col) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    rotate(&src, &mut dst, RotateCode::Rotate90Clockwise).unwrap();

    // Top-left of dst should be bottom-left of src
    assert_eq!(dst.at(0, 0).unwrap()[0], src.at(9, 0).unwrap()[0]);
    // Top-right of dst should be top-left of src
    assert_eq!(dst.at(0, 9).unwrap()[0], src.at(0, 0).unwrap()[0]);
}

#[test]
fn test_rotate_180_correctness() {
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();
    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = (row * 10 + col) as u8;
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    rotate(&src, &mut dst, RotateCode::Rotate180).unwrap();

    // Corners should be swapped diagonally
    assert_eq!(dst.at(0, 0).unwrap()[0], src.at(9, 9).unwrap()[0]);
    assert_eq!(dst.at(9, 9).unwrap()[0], src.at(0, 0).unwrap()[0]);
}

#[test]
#[ignore]
fn test_rotate_visual_inspection() {
    let mut src = Mat::new(5, 5, 1, MatDepth::U8).unwrap();
    for row in 0..5 {
        for col in 0..5 {
            src.at_mut(row, col).unwrap()[0] = (row * 10 + col) as u8;
        }
    }

    println!("\nOriginal:");
    print_image_data(&src, "Source", 5, 5);

    let mut dst_90cw = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    rotate(&src, &mut dst_90cw, RotateCode::Rotate90Clockwise).unwrap();
    println!("\n90° Clockwise:");
    print_image_data(&dst_90cw, "90CW", 5, 5);

    let mut dst_180 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    rotate(&src, &mut dst_180, RotateCode::Rotate180).unwrap();
    println!("\n180°:");
    print_image_data(&dst_180, "180", 5, 5);

    let mut dst_90ccw = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    rotate(&src, &mut dst_90ccw, RotateCode::Rotate90CounterClockwise).unwrap();
    println!("\n90° Counter-Clockwise:");
    print_image_data(&dst_90ccw, "90CCW", 5, 5);
}
