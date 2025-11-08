// Core module tests ported from OpenCV test suite
// opencv/modules/core/test/test_mat.cpp
// opencv/modules/core/test/test_operations.cpp

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::Scalar;

/// Test from test_mat.cpp - Mat creation and basic properties
#[test]
fn test_mat_creation_properties() {
    let mat = Mat::new(10, 20, 3, MatDepth::U8).unwrap();

    assert_eq!(mat.rows(), 10);
    assert_eq!(mat.cols(), 20);
    assert_eq!(mat.channels(), 3);
    assert_eq!(mat.depth(), MatDepth::U8);
}

/// Test Mat::new_with_default from test_mat.cpp
#[test]
fn test_mat_initialization_with_scalar() {
    let mat = Mat::new_with_default(5, 5, 3, MatDepth::U8, Scalar::all(42.0)).unwrap();

    // All pixels should be initialized to 42
    for row in 0..5 {
        for col in 0..5 {
            let pixel = mat.at(row, col).unwrap();
            for ch in 0..3 {
                assert_eq!(pixel[ch], 42, "Pixel at ({},{}) channel {} should be 42", row, col, ch);
            }
        }
    }
}

/// Test Mat cloning from test_mat.cpp
#[test]
fn test_mat_clone_creates_deep_copy() {
    let mut mat1 = Mat::new_with_default(3, 3, 1, MatDepth::U8, Scalar::all(100.0)).unwrap();
    let mat2 = mat1.clone();

    // Modify original
    mat1.at_mut(0, 0).unwrap()[0] = 200;

    // Clone should be unaffected
    assert_eq!(mat2.at(0, 0).unwrap()[0], 100, "Clone should not be affected by original modification");
    assert_eq!(mat1.at(0, 0).unwrap()[0], 200, "Original should be modified");
}

/// Test Mat depth conversion from test_mat.cpp
#[test]
fn test_mat_convert_depth_u8_to_f32() {
    let mut src = Mat::new(3, 3, 1, MatDepth::U8).unwrap();

    // Set test values
    src.at_mut(0, 0).unwrap()[0] = 0;
    src.at_mut(1, 1).unwrap()[0] = 128;
    src.at_mut(2, 2).unwrap()[0] = 255;

    let dst = src.convert_to(MatDepth::F32).unwrap();

    assert_eq!(dst.depth(), MatDepth::F32);

    // Values should be normalized to 0.0-1.0
    let v0 = dst.at_f32(0, 0, 0).unwrap();
    let v1 = dst.at_f32(1, 1, 0).unwrap();
    let v2 = dst.at_f32(2, 2, 0).unwrap();

    assert!((v0 - 0.0).abs() < 0.01, "0 should convert to ~0.0");
    assert!((v1 - 0.5).abs() < 0.1, "128 should convert to ~0.5");
    assert!((v2 - 1.0).abs() < 0.01, "255 should convert to ~1.0");
}

/// Test Mat depth conversion F32 to U8 from test_mat.cpp
#[test]
fn test_mat_convert_depth_f32_to_u8() {
    let mut src = Mat::new(3, 3, 1, MatDepth::F32).unwrap();

    // Set normalized values
    src.set_f32(0, 0, 0, 0.0).unwrap();
    src.set_f32(1, 1, 0, 0.5).unwrap();
    src.set_f32(2, 2, 0, 1.0).unwrap();

    let dst = src.convert_to(MatDepth::U8).unwrap();

    assert_eq!(dst.depth(), MatDepth::U8);

    let v0 = dst.at(0, 0).unwrap()[0];
    let v1 = dst.at(1, 1).unwrap()[0];
    let v2 = dst.at(2, 2).unwrap()[0];

    assert_eq!(v0, 0, "0.0 should convert to 0");
    assert!((v1 as i32 - 128).abs() <= 1, "0.5 should convert to ~128, got {}", v1);
    assert_eq!(v2, 255, "1.0 should convert to 255");
}

/// Test multi-channel Mat from test_mat.cpp
#[test]
fn test_mat_multi_channel_access() {
    let mut mat = Mat::new(5, 5, 3, MatDepth::U8).unwrap();

    // Set different values per channel
    mat.at_mut(2, 2).unwrap()[0] = 10;  // R
    mat.at_mut(2, 2).unwrap()[1] = 20;  // G
    mat.at_mut(2, 2).unwrap()[2] = 30;  // B

    let pixel = mat.at(2, 2).unwrap();
    assert_eq!(pixel[0], 10);
    assert_eq!(pixel[1], 20);
    assert_eq!(pixel[2], 30);
}

/// Test Mat bounds checking from test_mat.cpp
#[test]
#[should_panic]
fn test_mat_access_out_of_bounds_panics() {
    let mat = Mat::new(10, 10, 1, MatDepth::U8).unwrap();

    // This should panic
    let _ = mat.at(11, 5);
}

/// Test F32 Mat operations from test_mat.cpp
#[test]
fn test_mat_f32_operations() {
    let mut mat = Mat::new(4, 4, 1, MatDepth::F32).unwrap();

    // Set some floating point values
    mat.set_f32(0, 0, 0, 1.5).unwrap();
    mat.set_f32(1, 1, 0, 2.7).unwrap();
    mat.set_f32(2, 2, 0, -3.14).unwrap();

    // Retrieve and verify
    let v0 = mat.at_f32(0, 0, 0).unwrap();
    let v1 = mat.at_f32(1, 1, 0).unwrap();
    let v2 = mat.at_f32(2, 2, 0).unwrap();

    assert!((v0 - 1.5).abs() < 1e-6);
    assert!((v1 - 2.7).abs() < 1e-6);
    assert!((v2 - (-3.14)).abs() < 1e-6);
}

/// Test F64 Mat operations from test_mat.cpp
#[test]
fn test_mat_f64_operations() {
    let mut mat = Mat::new(3, 3, 1, MatDepth::F64).unwrap();

    // Set high precision values
    mat.set_f64(0, 0, 0, std::f64::consts::PI).unwrap();
    mat.set_f64(1, 1, 0, std::f64::consts::E).unwrap();

    let pi = mat.at_f64(0, 0, 0).unwrap();
    let e = mat.at_f64(1, 1, 0).unwrap();

    assert!((pi - std::f64::consts::PI).abs() < 1e-10);
    assert!((e - std::f64::consts::E).abs() < 1e-10);
}

/// Test U16 Mat operations from test_mat.cpp
#[test]
fn test_mat_u16_operations() {
    let mut mat = Mat::new(3, 3, 1, MatDepth::U16).unwrap();

    // U16 can store values 0-65535
    mat.set_u16(0, 0, 0, 0).unwrap();
    mat.set_u16(1, 1, 0, 32768).unwrap();
    mat.set_u16(2, 2, 0, 65535).unwrap();

    assert_eq!(mat.at_u16(0, 0, 0).unwrap(), 0);
    assert_eq!(mat.at_u16(1, 1, 0).unwrap(), 32768);
    assert_eq!(mat.at_u16(2, 2, 0).unwrap(), 65535);
}

/// Test Scalar operations from test_operations.cpp
#[test]
fn test_scalar_new() {
    let s1 = Scalar::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(s1.val[0], 1.0);
    assert_eq!(s1.val[1], 2.0);
    assert_eq!(s1.val[2], 3.0);
    assert_eq!(s1.val[3], 4.0);
}

/// Test Scalar::all from test_operations.cpp
#[test]
fn test_scalar_all() {
    let s = Scalar::all(42.0);
    assert_eq!(s.val[0], 42.0);
    assert_eq!(s.val[1], 42.0);
    assert_eq!(s.val[2], 42.0);
    assert_eq!(s.val[3], 42.0);
}

/// Test Mat with different channel counts from test_mat.cpp
#[test]
fn test_mat_channel_counts() {
    for &channels in &[1, 2, 3, 4] {
        let mat = Mat::new(5, 5, channels, MatDepth::U8).unwrap();
        assert_eq!(mat.channels(), channels, "Mat should have {} channels", channels);
    }
}

/// Test Mat row/col iteration from test_mat.cpp
#[test]
fn test_mat_iteration() {
    let mut mat = Mat::new(3, 4, 1, MatDepth::U8).unwrap();

    // Fill with row*cols + col
    for row in 0..3 {
        for col in 0..4 {
            mat.at_mut(row, col).unwrap()[0] = (row * 4 + col) as u8;
        }
    }

    // Verify
    for row in 0..3 {
        for col in 0..4 {
            let expected = (row * 4 + col) as u8;
            let actual = mat.at(row, col).unwrap()[0];
            assert_eq!(actual, expected, "Value at ({},{}) should be {}", row, col, expected);
        }
    }
}
