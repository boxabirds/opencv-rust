// Camera calibration tests ported from OpenCV test suite
// opencv/modules/calib3d/test/test_camera_calibration.cpp
// opencv/modules/calib3d/test/test_fisheye.cpp
// opencv/modules/calib3d/test/test_undistort.cpp

use opencv_rust::calib3d::camera::*;
use opencv_rust::calib3d::fisheye::*;
use opencv_rust::core::types::{Point, Point2f, Point3f};

/// Test from opencv test_camera_calibration.cpp - camera matrix creation
#[test]
fn test_camera_matrix_creation() {
    let camera = CameraMatrix::new(800.0, 800.0, 320.0, 240.0);

    assert_eq!(camera.fx, 800.0);
    assert_eq!(camera.fy, 800.0);
    assert_eq!(camera.cx, 320.0);
    assert_eq!(camera.cy, 240.0);
}

/// Test camera matrix to/from matrix conversion from opencv test_camera_calibration.cpp
#[test]
fn test_camera_matrix_conversion() {
    let original = CameraMatrix::new(600.0, 600.0, 320.0, 240.0);
    let matrix = original.to_matrix();

    // Check matrix structure
    assert_eq!(matrix[0][0], 600.0); // fx
    assert_eq!(matrix[1][1], 600.0); // fy
    assert_eq!(matrix[0][2], 320.0); // cx
    assert_eq!(matrix[1][2], 240.0); // cy
    assert_eq!(matrix[2][2], 1.0);
    assert_eq!(matrix[0][1], 0.0); // Skew should be 0
    assert_eq!(matrix[1][0], 0.0);
    assert_eq!(matrix[2][0], 0.0);
    assert_eq!(matrix[2][1], 0.0);

    // Round-trip conversion
    let converted = CameraMatrix::from_matrix(&matrix);
    assert_eq!(converted.fx, original.fx);
    assert_eq!(converted.fy, original.fy);
    assert_eq!(converted.cx, original.cx);
    assert_eq!(converted.cy, original.cy);
}

/// Test 3D to 2D projection from opencv test_camera_calibration.cpp
#[test]
fn test_camera_projection() {
    let camera = CameraMatrix::new(500.0, 500.0, 320.0, 240.0);

    // Project point directly in front of camera
    let point_3d = Point3f::new(0.0, 0.0, 1.0);
    let point_2d = camera.project(&point_3d);

    // Should project to principal point
    assert_eq!(point_2d.x, 320);
    assert_eq!(point_2d.y, 240);
}

/// Test projection with offset from opencv test_camera_calibration.cpp
#[test]
fn test_camera_projection_offset() {
    let camera = CameraMatrix::new(500.0, 500.0, 320.0, 240.0);

    // Point to the right and above optical axis
    let point_3d = Point3f::new(0.5, 0.5, 1.0);
    let point_2d = camera.project(&point_3d);

    // Should project to right and above principal point
    // x = fx * (X/Z) + cx = 500 * 0.5 + 320 = 570
    // y = fy * (Y/Z) + cy = 500 * 0.5 + 240 = 490
    assert_eq!(point_2d.x, 570);
    assert_eq!(point_2d.y, 490);
}

/// Test unprojection from opencv test_camera_calibration.cpp
#[test]
fn test_camera_unprojection() {
    let camera = CameraMatrix::new(500.0, 500.0, 320.0, 240.0);

    // Unproject principal point at depth 1
    let point_2d = Point::new(320, 240);
    let point_3d = camera.unproject(&point_2d, 1.0);

    // Should get point at (0, 0, 1)
    assert!((point_3d.x - 0.0).abs() < 0.01);
    assert!((point_3d.y - 0.0).abs() < 0.01);
    assert!((point_3d.z - 1.0).abs() < 0.01);
}

/// Test projection/unprojection round-trip from opencv test_camera_calibration.cpp
#[test]
fn test_camera_project_unproject_roundtrip() {
    let camera = CameraMatrix::new(500.0, 500.0, 320.0, 240.0);

    let original_3d = Point3f::new(0.3, -0.2, 2.0);
    let projected_2d = camera.project(&original_3d);
    let unprojected_3d = camera.unproject(&projected_2d, 2.0);

    // Should recover original point
    assert!((unprojected_3d.x - original_3d.x).abs() < 0.1);
    assert!((unprojected_3d.y - original_3d.y).abs() < 0.1);
    assert!((unprojected_3d.z - original_3d.z).abs() < 0.01);
}

/// Test distortion coefficient creation from opencv test_undistort.cpp
#[test]
fn test_distortion_coefficients() {
    let dist = DistortionCoefficients::new(-0.2, 0.1, -0.05, 0.01, -0.01);

    assert_eq!(dist.k[0], -0.2);
    assert_eq!(dist.k[1], 0.1);
    assert_eq!(dist.k[2], -0.05);
    assert_eq!(dist.p[0], 0.01);
    assert_eq!(dist.p[1], -0.01);
}

/// Test zero distortion from opencv test_undistort.cpp
#[test]
fn test_zero_distortion_identity() {
    let dist = DistortionCoefficients::zero();

    // Zero distortion should not change coordinates
    let (x, y) = (0.5, 0.3);
    let (x_dist, y_dist) = dist.distort(x, y);

    assert!((x_dist - x).abs() < 1e-6);
    assert!((y_dist - y).abs() < 1e-6);
}

/// Test radial distortion from opencv test_undistort.cpp
#[test]
fn test_radial_distortion() {
    // Negative k1 causes barrel distortion
    let dist = DistortionCoefficients::new(-0.2, 0.0, 0.0, 0.0, 0.0);

    let (x, y) = (0.5, 0.0);
    let (x_dist, _y_dist) = dist.distort(x, y);

    // Barrel distortion should move point outward
    // For negative k1, radial factor < 1, so point moves inward actually
    // r2 = 0.25, radial = 1 + (-0.2)*0.25 = 0.95
    // x_dist = 0.5 * 0.95 = 0.475
    assert!((x_dist - 0.475).abs() < 0.01, "Expected barrel distortion effect");
}

/// Test distortion/undistortion round-trip from opencv test_undistort.cpp
#[test]
fn test_distort_undistort_roundtrip() {
    let dist = DistortionCoefficients::new(-0.1, 0.05, 0.0, 0.0, 0.0);

    let (x, y) = (0.3, 0.4);
    let (x_dist, y_dist) = dist.distort(x, y);
    let (x_undist, y_undist) = dist.undistort(x_dist, y_dist);

    // Should recover original coordinates
    assert!((x_undist - x).abs() < 0.01, "X coordinate should be recovered");
    assert!((y_undist - y).abs() < 0.01, "Y coordinate should be recovered");
}

/// Test fisheye distortion creation from opencv test_fisheye.cpp
#[test]
fn test_fisheye_distortion_creation() {
    let dist = FisheyeDistortion::new();

    assert_eq!(dist.k1, 0.0);
    assert_eq!(dist.k2, 0.0);
    assert_eq!(dist.k3, 0.0);
    assert_eq!(dist.k4, 0.0);
}

/// Test fisheye distortion from array from opencv test_fisheye.cpp
#[test]
fn test_fisheye_from_array() {
    let coeffs = [0.1, -0.2, 0.05, -0.01];
    let dist = FisheyeDistortion::from_array(&coeffs);

    assert_eq!(dist.k1, 0.1);
    assert_eq!(dist.k2, -0.2);
    assert_eq!(dist.k3, 0.05);
    assert_eq!(dist.k4, -0.01);

    // Test round-trip
    let array = dist.to_array();
    assert_eq!(array, coeffs);
}

/// Test fisheye camera matrix from opencv test_fisheye.cpp
#[test]
fn test_fisheye_camera_matrix() {
    let camera = FisheyeCameraMatrix::new(400.0, 400.0, 320.0, 240.0);

    assert_eq!(camera.fx, 400.0);
    assert_eq!(camera.fy, 400.0);
    assert_eq!(camera.cx, 320.0);
    assert_eq!(camera.cy, 240.0);

    let matrix = camera.to_matrix();
    assert_eq!(matrix[0][0], 400.0);
    assert_eq!(matrix[1][1], 400.0);
    assert_eq!(matrix[0][2], 320.0);
    assert_eq!(matrix[1][2], 240.0);
}

/// Test fisheye projection with zero distortion from opencv test_fisheye.cpp
#[test]
fn test_fisheye_project_zero_distortion() {
    let camera = FisheyeCameraMatrix::new(500.0, 500.0, 320.0, 240.0);
    let distortion = FisheyeDistortion::new();
    let rvec = [0.0, 0.0, 0.0]; // No rotation
    let tvec = [0.0, 0.0, 0.0]; // No translation

    // Point in front of camera
    let object_points = vec![Point3f::new(0.0, 0.0, 1.0)];

    let image_points = fisheye_project_points(
        &object_points,
        &camera,
        &distortion,
        &rvec,
        &tvec,
    ).unwrap();

    assert_eq!(image_points.len(), 1);

    // Should project to principal point
    assert!((image_points[0].x - 320.0).abs() < 1.0);
    assert!((image_points[0].y - 240.0).abs() < 1.0);
}

/// Test fisheye projection with multiple points from opencv test_fisheye.cpp
#[test]
fn test_fisheye_project_multiple_points() {
    let camera = FisheyeCameraMatrix::new(400.0, 400.0, 200.0, 150.0);
    let distortion = FisheyeDistortion::from_array(&[0.0, 0.0, 0.0, 0.0]);
    let rvec = [0.0, 0.0, 0.0];
    let tvec = [0.0, 0.0, 0.0];

    let object_points = vec![
        Point3f::new(0.0, 0.0, 1.0),
        Point3f::new(0.5, 0.5, 1.0),
        Point3f::new(-0.5, -0.5, 1.0),
    ];

    let image_points = fisheye_project_points(
        &object_points,
        &camera,
        &distortion,
        &rvec,
        &tvec,
    ).unwrap();

    assert_eq!(image_points.len(), 3);

    // Check that points are ordered correctly
    // Center point should be at principal point
    assert!((image_points[0].x - 200.0).abs() < 2.0);
    assert!((image_points[0].y - 150.0).abs() < 2.0);
}

/// Test fisheye projection error cases from opencv test_fisheye.cpp
#[test]
fn test_fisheye_project_empty_points() {
    let camera = FisheyeCameraMatrix::new(400.0, 400.0, 200.0, 150.0);
    let distortion = FisheyeDistortion::new();
    let rvec = [0.0, 0.0, 0.0];
    let tvec = [0.0, 0.0, 0.0];

    let object_points = vec![];

    let result = fisheye_project_points(
        &object_points,
        &camera,
        &distortion,
        &rvec,
        &tvec,
    );

    assert!(result.is_err(), "Should reject empty point array");
}

/// Test fisheye undistortion from opencv test_fisheye.cpp
#[test]
fn test_fisheye_undistort_points() {
    let camera = FisheyeCameraMatrix::new(400.0, 400.0, 320.0, 240.0);
    let distortion = FisheyeDistortion::new(); // Zero distortion

    let distorted_points = vec![
        Point2f::new(320.0, 240.0),
        Point2f::new(420.0, 340.0),
    ];

    let undistorted = fisheye_undistort_points(
        &distorted_points,
        &camera,
        &distortion,
    ).unwrap();

    assert_eq!(undistorted.len(), 2);
}

/// Test fisheye undistortion empty points from opencv test_fisheye.cpp
#[test]
fn test_fisheye_undistort_empty() {
    let camera = FisheyeCameraMatrix::new(400.0, 400.0, 320.0, 240.0);
    let distortion = FisheyeDistortion::new();

    let distorted_points = vec![];

    let result = fisheye_undistort_points(
        &distorted_points,
        &camera,
        &distortion,
    );

    assert!(result.is_err(), "Should reject empty points");
}

/// Test fisheye equidistant model from opencv test_fisheye.cpp
#[test]
fn test_fisheye_equidistant_model() {
    let camera = FisheyeCameraMatrix::new(300.0, 300.0, 320.0, 240.0);

    // Strong fisheye distortion
    let distortion = FisheyeDistortion::from_array(&[0.1, -0.05, 0.01, -0.001]);

    let rvec = [0.0, 0.0, 0.0];
    let tvec = [0.0, 0.0, 0.0];

    // Points at different distances from optical axis
    let object_points = vec![
        Point3f::new(0.0, 0.0, 1.0),    // Center
        Point3f::new(0.5, 0.0, 1.0),    // Moderate angle
        Point3f::new(1.0, 0.0, 1.0),    // Large angle
    ];

    let image_points = fisheye_project_points(
        &object_points,
        &camera,
        &distortion,
        &rvec,
        &tvec,
    ).unwrap();

    assert_eq!(image_points.len(), 3);

    // Center should be close to principal point
    assert!((image_points[0].x - 320.0).abs() < 5.0);

    // Points should be on the x-axis (y should be close to cy)
    for point in &image_points {
        assert!((point.y - 240.0).abs() < 10.0);
    }

    // Distortion should affect distance from center non-linearly
    let dist1 = (image_points[1].x - 320.0).abs();
    let dist2 = (image_points[2].x - 320.0).abs();

    // Second point should be further from center
    assert!(dist2 > dist1, "Fisheye distortion should increase with angle");
}

/// Test camera projection with different depths from opencv test_camera_calibration.cpp
#[test]
fn test_camera_projection_depth_scaling() {
    let camera = CameraMatrix::new(500.0, 500.0, 320.0, 240.0);

    let point_near = Point3f::new(1.0, 1.0, 1.0);
    let point_far = Point3f::new(2.0, 2.0, 2.0);

    let proj_near = camera.project(&point_near);
    let proj_far = camera.project(&point_far);

    // Both should project to same 2D point (perspective projection)
    assert_eq!(proj_near.x, proj_far.x);
    assert_eq!(proj_near.y, proj_far.y);
}
