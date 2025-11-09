// Video analysis tests ported from OpenCV test suite
// opencv/modules/video/test/test_optflowpyrlk.cpp
// opencv/modules/video/test/test_OF_accuracy.cpp
// opencv/modules/video/test/test_bgfg.cpp

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::{Point, Size, Scalar};
use opencv_rust::video::optical_flow::*;
use opencv_rust::video::background_subtraction::*;

/// Create moving pattern for optical flow testing
fn create_shifted_image(original: &Mat, dx: i32, dy: i32) -> Mat {
    let mut shifted = Mat::new(original.rows(), original.cols(), 1, MatDepth::U8).unwrap();

    for row in 0..original.rows() {
        for col in 0..original.cols() {
            let src_row = (row as i32 - dy).max(0).min(original.rows() as i32 - 1) as usize;
            let src_col = (col as i32 - dx).max(0).min(original.cols() as i32 - 1) as usize;

            shifted.at_mut(row, col).unwrap()[0] = original.at(src_row, src_col).unwrap()[0];
        }
    }

    shifted
}

/// Test from opencv test_optflowpyrlk.cpp - basic tracking
#[test]
fn test_optical_flow_lk_basic() {
    // Create first image with a bright spot
    let mut img1 = Mat::new(100, 100, 1, MatDepth::U8).unwrap();
    for row in 0..100 {
        for col in 0..100 {
            img1.at_mut(row, col).unwrap()[0] = 50;
        }
    }

    // Add a bright feature
    for row in 40..60 {
        for col in 40..60 {
            img1.at_mut(row, col).unwrap()[0] = 200;
        }
    }

    // Create second image - shift the bright spot
    let img2 = create_shifted_image(&img1, 10, 5);

    // Track the center point
    let prev_pts = vec![Point::new(50, 50)];

    let (next_pts, status) = calc_optical_flow_pyr_lk(
        &img1,
        &img2,
        &prev_pts,
        Size::new(15, 15),
        3
    ).unwrap();

    assert_eq!(next_pts.len(), 1);
    assert_eq!(status.len(), 1);

    // Status should indicate successful tracking
    assert_eq!(status[0], 1, "Point should be tracked successfully");

    // Point should move in direction of shift
    let moved_x = next_pts[0].x - prev_pts[0].x;
    let moved_y = next_pts[0].y - prev_pts[0].y;

    // Should detect movement in positive x direction
    assert!(moved_x > 0, "Should detect rightward movement, got dx={}", moved_x);
    assert!(moved_y > 0, "Should detect downward movement, got dy={}", moved_y);
}

/// Test optical flow with no motion from opencv test_optflowpyrlk.cpp
#[test]
fn test_optical_flow_lk_no_motion() {
    let mut img = Mat::new(80, 80, 1, MatDepth::U8).unwrap();

    // Create checkerboard pattern
    for row in 0..80 {
        for col in 0..80 {
            let val = if (row / 10 + col / 10) % 2 == 0 { 0 } else { 200 };
            img.at_mut(row, col).unwrap()[0] = val;
        }
    }

    let prev_pts = vec![
        Point::new(40, 40),
        Point::new(20, 20),
        Point::new(60, 60),
    ];

    // Same image - no motion
    let (next_pts, status) = calc_optical_flow_pyr_lk(
        &img,
        &img,
        &prev_pts,
        Size::new(15, 15),
        3
    ).unwrap();

    // All points should be tracked
    for &s in &status {
        assert_eq!(s, 1, "Points should be tracked in static scene");
    }

    // Points should stay in place (or very close)
    for i in 0..prev_pts.len() {
        let dx = (next_pts[i].x - prev_pts[i].x).abs();
        let dy = (next_pts[i].y - prev_pts[i].y).abs();

        assert!(
            dx <= 2 && dy <= 2,
            "Point should not move significantly: dx={}, dy={}",
            dx, dy
        );
    }
}

/// Test optical flow requires grayscale from opencv test_optflowpyrlk.cpp
#[test]
fn test_optical_flow_requires_grayscale() {
    let color_img = Mat::new(50, 50, 3, MatDepth::U8).unwrap();
    let gray_img = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    let pts = vec![Point::new(25, 25)];

    let result = calc_optical_flow_pyr_lk(
        &color_img,
        &gray_img,
        &pts,
        Size::new(15, 15),
        3
    );

    assert!(result.is_err(), "Should reject color images");
}

/// Test optical flow with border points from opencv test_optflowpyrlk.cpp
#[test]
fn test_optical_flow_border_handling() {
    let img1 = Mat::new(50, 50, 1, MatDepth::U8).unwrap();
    let img2 = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    // Points too close to border
    let border_pts = vec![
        Point::new(2, 2),
        Point::new(47, 47),
        Point::new(2, 47),
    ];

    let (_next_pts, status) = calc_optical_flow_pyr_lk(
        &img1,
        &img2,
        &border_pts,
        Size::new(15, 15),
        3
    ).unwrap();

    // Border points should be marked as failed
    for &s in &status {
        assert_eq!(s, 0, "Border points should fail tracking");
    }
}

/// Test optical flow with multiple points from opencv test_optflowpyrlk.cpp
#[test]
fn test_optical_flow_multiple_points() {
    let mut img = Mat::new(100, 100, 1, MatDepth::U8).unwrap();

    // Create gradient
    for row in 0..100 {
        for col in 0..100 {
            img.at_mut(row, col).unwrap()[0] = ((row + col) % 256) as u8;
        }
    }

    let prev_pts = vec![
        Point::new(30, 30),
        Point::new(50, 50),
        Point::new(70, 70),
    ];

    let (next_pts, status) = calc_optical_flow_pyr_lk(
        &img,
        &img,
        &prev_pts,
        Size::new(15, 15),
        3
    ).unwrap();

    assert_eq!(next_pts.len(), prev_pts.len());
    assert_eq!(status.len(), prev_pts.len());
}

/// Test Farneback optical flow from opencv test_OF_accuracy.cpp
#[test]
fn test_farneback_optical_flow_basic() {
    let img1 = Mat::new(64, 64, 1, MatDepth::U8).unwrap();
    let img2 = Mat::new(64, 64, 1, MatDepth::U8).unwrap();

    let flow = calc_optical_flow_farneback(
        &img1,
        &img2,
        0.5,  // pyr_scale
        3,    // levels
        15,   // winsize
        3     // iterations
    ).unwrap();

    // Flow should be 2-channel (x, y flow)
    assert_eq!(flow.channels(), 2);
    assert_eq!(flow.rows(), img1.rows());
    assert_eq!(flow.cols(), img1.cols());
}

/// Test Farneback requires grayscale from opencv test_OF_accuracy.cpp
#[test]
fn test_farneback_requires_grayscale() {
    let color_img = Mat::new(50, 50, 3, MatDepth::U8).unwrap();
    let gray_img = Mat::new(50, 50, 1, MatDepth::U8).unwrap();

    let result = calc_optical_flow_farneback(&color_img, &gray_img, 0.5, 3, 15, 3);

    assert!(result.is_err(), "Farneback should require grayscale");
}

/// Test Farneback with mismatched dimensions from opencv test_OF_accuracy.cpp
#[test]
fn test_farneback_dimension_mismatch() {
    let img1 = Mat::new(50, 50, 1, MatDepth::U8).unwrap();
    let img2 = Mat::new(60, 60, 1, MatDepth::U8).unwrap();

    let result = calc_optical_flow_farneback(&img1, &img2, 0.5, 3, 15, 3);

    assert!(result.is_err(), "Should reject mismatched dimensions");
}

/// Test MOG2 background subtractor from opencv test_bgfg.cpp
#[test]
fn test_mog2_basic() {
    let mut mog2 = BackgroundSubtractorMOG2::new();

    // Create RGB image
    let img = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::new(128.0, 128.0, 128.0, 0.0)).unwrap();
    let mut fgmask = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    let result = mog2.apply(&img, &mut fgmask, -1.0);

    assert!(result.is_ok(), "MOG2 should process successfully");
    assert_eq!(fgmask.rows(), img.rows());
    assert_eq!(fgmask.cols(), img.cols());
    assert_eq!(fgmask.channels(), 1);
}

/// Test MOG2 with multiple frames from opencv test_bgfg.cpp
#[test]
fn test_mog2_multiple_frames() {
    let mut mog2 = BackgroundSubtractorMOG2::with_params(100, 16.0, true);

    // Process several background frames
    let bg = Mat::new_with_default(60, 60, 3, MatDepth::U8, Scalar::new(100.0, 100.0, 100.0, 0.0)).unwrap();
    let mut fgmask = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    for _ in 0..10 {
        mog2.apply(&bg, &mut fgmask, -1.0).unwrap();
    }

    // After learning background, uniform image should be mostly background
    let mut background_pixels = 0;
    for row in 0..fgmask.rows() {
        for col in 0..fgmask.cols() {
            if fgmask.at(row, col).unwrap()[0] == 0 {
                background_pixels += 1;
            }
        }
    }

    let total_pixels = fgmask.rows() * fgmask.cols();
    let background_ratio = background_pixels as f64 / total_pixels as f64;

    assert!(
        background_ratio > 0.5,
        "After learning, background should be detected: {:.2}%",
        background_ratio * 100.0
    );
}

/// Test MOG2 detects foreground from opencv test_bgfg.cpp
#[test]
fn test_mog2_foreground_detection() {
    let mut mog2 = BackgroundSubtractorMOG2::new();
    let mut fgmask = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    // Learn static background
    let bg = Mat::new_with_default(80, 80, 3, MatDepth::U8, Scalar::new(50.0, 50.0, 50.0, 0.0)).unwrap();

    for _ in 0..20 {
        mog2.apply(&bg, &mut fgmask, -1.0).unwrap();
    }

    // Add foreground object
    let mut fg = Mat::new_with_default(80, 80, 3, MatDepth::U8, Scalar::new(50.0, 50.0, 50.0, 0.0)).unwrap();

    // Add bright square in center
    for row in 30..50 {
        for col in 30..50 {
            let pixel = fg.at_mut(row, col).unwrap();
            pixel[0] = 200;
            pixel[1] = 200;
            pixel[2] = 200;
        }
    }

    mog2.apply(&fg, &mut fgmask, -1.0).unwrap();

    // Count foreground pixels in the bright square region
    let mut fg_pixels = 0;
    for row in 35..45 {
        for col in 35..45 {
            if fgmask.at(row, col).unwrap()[0] > 0 {
                fg_pixels += 1;
            }
        }
    }

    assert!(
        fg_pixels > 20,
        "Should detect foreground object, found {} fg pixels",
        fg_pixels
    );
}

/// Test MOG2 requires 3-channel image from opencv test_bgfg.cpp
#[test]
fn test_mog2_requires_three_channels() {
    let mut mog2 = BackgroundSubtractorMOG2::new();
    let gray = Mat::new(50, 50, 1, MatDepth::U8).unwrap();
    let mut fgmask = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    let result = mog2.apply(&gray, &mut fgmask, -1.0);

    assert!(result.is_err(), "MOG2 should require 3-channel image");
}

/// Test MOG2 with custom parameters from opencv test_bgfg.cpp
#[test]
fn test_mog2_custom_parameters() {
    let mog2 = BackgroundSubtractorMOG2::with_params(200, 25.0, false);

    assert_eq!(mog2.history, 200);
    assert_eq!(mog2.var_threshold, 25.0);
    assert_eq!(mog2.detect_shadows, false);
}

/// Test KNN background subtractor from opencv test_bgfg.cpp
#[test]
fn test_knn_basic() {
    let mut knn = BackgroundSubtractorKNN::new();

    let img = Mat::new_with_default(40, 40, 3, MatDepth::U8, Scalar::new(100.0, 100.0, 100.0, 0.0)).unwrap();
    let mut fgmask = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    let result = knn.apply(&img, &mut fgmask, -1.0);

    assert!(result.is_ok(), "KNN should process successfully");
    assert_eq!(fgmask.rows(), img.rows());
    assert_eq!(fgmask.cols(), img.cols());
}

/// Test KNN with multiple frames from opencv test_bgfg.cpp
#[test]
fn test_knn_learning() {
    let mut knn = BackgroundSubtractorKNN::with_params(100, 400.0, true);
    let mut fgmask = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    // Learn background
    let bg = Mat::new_with_default(50, 50, 3, MatDepth::U8, Scalar::new(75.0, 75.0, 75.0, 0.0)).unwrap();

    for _ in 0..15 {
        knn.apply(&bg, &mut fgmask, -1.0).unwrap();
    }

    // After learning, background should be detected
    let mut background_count = 0;
    for row in 0..fgmask.rows() {
        for col in 0..fgmask.cols() {
            if fgmask.at(row, col).unwrap()[0] == 0 {
                background_count += 1;
            }
        }
    }

    let total = fgmask.rows() * fgmask.cols();
    assert!(
        background_count as f64 / total as f64 > 0.3,
        "KNN should learn background"
    );
}
