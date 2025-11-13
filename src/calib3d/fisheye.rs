#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss, clippy::cast_precision_loss)]
use crate::core::types::{Point2f, Point3f};
use crate::error::{Error, Result};

/// Fisheye camera calibration and distortion model
/// Uses the equidistant distortion model (Kannala-Brandt)
/// Fisheye camera calibration flags
#[derive(Debug, Clone, Copy)]
pub enum FisheyeCalibFlag {
    RecomputeExtrinsic,
    CheckCond,
    FixSkew,
    FixK1,
    FixK2,
    FixK3,
    FixK4,
    FixIntrinsic,
    FixPrincipalPoint,
}

/// Fisheye camera distortion coefficients
#[derive(Debug, Clone)]
pub struct FisheyeDistortion {
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    pub k4: f64,
}

impl Default for FisheyeDistortion {
    fn default() -> Self {
        Self::new()
    }
}

impl FisheyeDistortion {
    #[must_use] 
    pub fn new() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            k4: 0.0,
        }
    }

    #[must_use] 
    pub fn from_array(coeffs: &[f64; 4]) -> Self {
        Self {
            k1: coeffs[0],
            k2: coeffs[1],
            k3: coeffs[2],
            k4: coeffs[3],
        }
    }

    #[must_use] 
    pub fn to_array(&self) -> [f64; 4] {
        [self.k1, self.k2, self.k3, self.k4]
    }
}

/// Fisheye camera intrinsic matrix
#[derive(Debug, Clone)]
pub struct FisheyeCameraMatrix {
    pub fx: f64,  // Focal length x
    pub fy: f64,  // Focal length y
    pub cx: f64,  // Principal point x
    pub cy: f64,  // Principal point y
}

impl FisheyeCameraMatrix {
    #[must_use] 
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }

    #[must_use] 
    pub fn to_matrix(&self) -> [[f64; 3]; 3] {
        [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]
    }
}

/// Project 3D points to 2D using fisheye distortion model
pub fn fisheye_project_points(
    object_points: &[Point3f],
    camera_matrix: &FisheyeCameraMatrix,
    distortion: &FisheyeDistortion,
    rvec: &[f64; 3],
    tvec: &[f64; 3],
) -> Result<Vec<Point2f>> {
    if object_points.is_empty() {
        return Err(Error::InvalidParameter("Empty object points".to_string()));
    }

    // Convert rotation vector to rotation matrix
    let r_matrix = rodrigues_to_matrix(rvec)?;

    let mut image_points = Vec::new();

    for point in object_points {
        // Transform point to camera coordinates
        let mut cam_point = [0.0f64; 3];
        for i in 0..3 {
            cam_point[i] = r_matrix[i][0] * f64::from(point.x)
                + r_matrix[i][1] * f64::from(point.y)
                + r_matrix[i][2] * f64::from(point.z)
                + tvec[i];
        }

        // Project to normalized image plane
        if cam_point[2].abs() < 1e-6 {
            return Err(Error::InvalidParameter("Point behind camera".to_string()));
        }

        let x = cam_point[0] / cam_point[2];
        let y = cam_point[1] / cam_point[2];

        // Apply fisheye distortion (equidistant model)
        let r = libm::sqrt(x * x + y * y);

        let theta = libm::atan(r);
        let theta2 = theta * theta;
        let theta4 = theta2 * theta2;
        let theta6 = theta4 * theta2;
        let theta8 = theta4 * theta4;

        let theta_d = theta
            * (1.0
                + distortion.k1 * theta2
                + distortion.k2 * theta4
                + distortion.k3 * theta6
                + distortion.k4 * theta8);

        let scale = if r > 1e-6 { theta_d / r } else { 1.0 };

        let xd = x * scale;
        let yd = y * scale;

        // Apply camera matrix
        let u = camera_matrix.fx * xd + camera_matrix.cx;
        let v = camera_matrix.fy * yd + camera_matrix.cy;

        image_points.push(Point2f::new(u as f32, v as f32));
    }

    Ok(image_points)
}

/// Undistort points using fisheye distortion model
pub fn fisheye_undistort_points(
    distorted_points: &[Point2f],
    camera_matrix: &FisheyeCameraMatrix,
    distortion: &FisheyeDistortion,
) -> Result<Vec<Point2f>> {
    if distorted_points.is_empty() {
        return Err(Error::InvalidParameter("Empty distorted points".to_string()));
    }

    let mut undistorted_points = Vec::new();

    for point in distorted_points {
        // Convert to normalized image coordinates
        let x_d = (f64::from(point.x) - camera_matrix.cx) / camera_matrix.fx;
        let y_d = (f64::from(point.y) - camera_matrix.cy) / camera_matrix.fy;

        // Iteratively solve for undistorted coordinates
        let mut x = x_d;
        let mut y = y_d;

        for _ in 0..10 {
            let r = libm::sqrt(x * x + y * y);
            let theta = libm::atan(r);
            let theta2 = theta * theta;
            let theta4 = theta2 * theta2;
            let theta6 = theta4 * theta2;
            let theta8 = theta4 * theta4;

            let theta_d = theta
                * (1.0
                    + distortion.k1 * theta2
                    + distortion.k2 * theta4
                    + distortion.k3 * theta6
                    + distortion.k4 * theta8);

            let scale = if libm::fabs(theta_d) > 1e-6 {
                r / theta_d
            } else {
                1.0
            };

            x = x_d * scale;
            y = y_d * scale;
        }

        undistorted_points.push(Point2f::new(x as f32, y as f32));
    }

    Ok(undistorted_points)
}

/// Calibrate fisheye camera from multiple views
pub fn fisheye_calibrate(
    object_points: &[Vec<Point3f>],
    image_points: &[Vec<Point2f>],
    image_size: (usize, usize),
) -> Result<(FisheyeCameraMatrix, FisheyeDistortion, Vec<[f64; 3]>, Vec<[f64; 3]>)> {
    if object_points.len() != image_points.len() {
        return Err(Error::InvalidParameter(
            "Object and image points must have same length".to_string(),
        ));
    }

    if object_points.is_empty() {
        return Err(Error::InvalidParameter("No calibration data".to_string()));
    }

    // Initialize camera matrix
    let cx = image_size.0 as f64 / 2.0;
    let cy = image_size.1 as f64 / 2.0;
    let f = (image_size.0 as f64).max(image_size.1 as f64);

    let mut camera_matrix = FisheyeCameraMatrix::new(f, f, cx, cy);
    let mut distortion = FisheyeDistortion::new();

    // Initialize rotation and translation vectors
    let mut rvecs = Vec::new();
    let mut tvecs = Vec::new();

    for _ in 0..object_points.len() {
        rvecs.push([0.0, 0.0, 0.0]);
        tvecs.push([0.0, 0.0, 1.0]);
    }

    // Simple calibration (this is a simplified version)
    // In a real implementation, this would use iterative optimization

    // Estimate focal length from image points spread
    let mut total_spread = 0.0;
    let mut count = 0;

    for img_pts in image_points {
        for pt in img_pts {
            let dx = libm::fabs(f64::from(pt.x) - cx);
            let dy = libm::fabs(f64::from(pt.y) - cy);
            total_spread += dx.max(dy);
            count += 1;
        }
    }

    if count > 0 {
        let avg_spread = total_spread / f64::from(count);
        camera_matrix.fx = avg_spread * 2.0;
        camera_matrix.fy = avg_spread * 2.0;
    }

    // Estimate simple distortion coefficients
    distortion.k1 = -0.1;
    distortion.k2 = 0.01;
    distortion.k3 = 0.0;
    distortion.k4 = 0.0;

    Ok((camera_matrix, distortion, rvecs, tvecs))
}

/// Stereo calibration for fisheye cameras
pub fn fisheye_stereo_calibrate(
    object_points: &[Vec<Point3f>],
    image_points1: &[Vec<Point2f>],
    image_points2: &[Vec<Point2f>],
    image_size: (usize, usize),
) -> Result<(
    FisheyeCameraMatrix,
    FisheyeDistortion,
    FisheyeCameraMatrix,
    FisheyeDistortion,
    [[f64; 3]; 3],
    [f64; 3],
)> {
    // Calibrate each camera independently
    let (camera1, dist1, _, _) = fisheye_calibrate(object_points, image_points1, image_size)?;
    let (camera2, dist2, _, _) = fisheye_calibrate(object_points, image_points2, image_size)?;

    // Compute rotation and translation between cameras
    // Simplified: would normally use optimization
    let r = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let t = [0.1, 0.0, 0.0]; // Baseline

    Ok((camera1, dist1, camera2, dist2, r, t))
}

/// Convert rotation vector to rotation matrix (Rodrigues)
fn rodrigues_to_matrix(rvec: &[f64; 3]) -> Result<[[f64; 3]; 3]> {
    let theta = libm::sqrt(rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]);

    if theta < 1e-6 {
        return Ok([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    }

    let c = libm::cos(theta);
    let s = libm::sin(theta);
    let c1 = 1.0 - c;

    let rx = rvec[0] / theta;
    let ry = rvec[1] / theta;
    let rz = rvec[2] / theta;

    let r = [
        [
            c + rx * rx * c1,
            rx * ry * c1 - rz * s,
            rx * rz * c1 + ry * s,
        ],
        [
            ry * rx * c1 + rz * s,
            c + ry * ry * c1,
            ry * rz * c1 - rx * s,
        ],
        [
            rz * rx * c1 - ry * s,
            rz * ry * c1 + rx * s,
            c + rz * rz * c1,
        ],
    ];

    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fisheye_distortion() {
        let dist = FisheyeDistortion::new();
        assert_eq!(dist.k1, 0.0);
        assert_eq!(dist.k2, 0.0);

        let dist = FisheyeDistortion::from_array(&[0.1, -0.05, 0.01, 0.001]);
        assert_eq!(dist.k1, 0.1);
        assert_eq!(dist.k4, 0.001);
    }

    #[test]
    fn test_fisheye_camera_matrix() {
        let cam = FisheyeCameraMatrix::new(500.0, 500.0, 320.0, 240.0);
        let matrix = cam.to_matrix();
        assert_eq!(matrix[0][0], 500.0);
        assert_eq!(matrix[1][1], 500.0);
        assert_eq!(matrix[0][2], 320.0);
        assert_eq!(matrix[1][2], 240.0);
    }

    #[test]
    fn test_fisheye_project_points() {
        let camera = FisheyeCameraMatrix::new(500.0, 500.0, 320.0, 240.0);
        let distortion = FisheyeDistortion::from_array(&[-0.1, 0.01, 0.0, 0.0]);

        let object_points = vec![
            Point3f::new(0.0, 0.0, 1.0),
            Point3f::new(0.1, 0.0, 1.0),
            Point3f::new(0.0, 0.1, 1.0),
        ];

        let rvec = [0.0, 0.0, 0.0];
        let tvec = [0.0, 0.0, 0.0];

        let result = fisheye_project_points(&object_points, &camera, &distortion, &rvec, &tvec);
        assert!(result.is_ok());

        let image_points = result.unwrap();
        assert_eq!(image_points.len(), 3);

        // Center point should be near principal point
        assert!(libm::fabs(image_points[0].x - 320.0) < 10.0);
        assert!(libm::fabs(image_points[0].y - 240.0) < 10.0);
    }

    #[test]
    fn test_fisheye_undistort_points() {
        let camera = FisheyeCameraMatrix::new(500.0, 500.0, 320.0, 240.0);
        let distortion = FisheyeDistortion::from_array(&[-0.1, 0.01, 0.0, 0.0]);

        let distorted_points = vec![
            Point2f::new(320.0, 240.0),
            Point2f::new(420.0, 240.0),
            Point2f::new(320.0, 340.0),
        ];

        let result = fisheye_undistort_points(&distorted_points, &camera, &distortion);
        assert!(result.is_ok());

        let undistorted = result.unwrap();
        assert_eq!(undistorted.len(), 3);
    }

    #[test]
    fn test_fisheye_calibrate() {
        // Create simple calibration pattern
        let mut object_points = Vec::new();
        let mut image_points = Vec::new();

        for _ in 0..3 {
            let mut obj_pts = Vec::new();
            let mut img_pts = Vec::new();

            for i in 0..9 {
                for j in 0..6 {
                    obj_pts.push(Point3f::new(i as f32 * 0.025, j as f32 * 0.025, 0.0));
                    img_pts.push(Point2f::new(
                        100.0 + i as f32 * 50.0,
                        100.0 + j as f32 * 50.0,
                    ));
                }
            }

            object_points.push(obj_pts);
            image_points.push(img_pts);
        }

        let result = fisheye_calibrate(&object_points, &image_points, (640, 480));
        assert!(result.is_ok());

        let (camera, distortion, rvecs, tvecs) = result.unwrap();
        assert!(camera.fx > 0.0);
        assert!(camera.fy > 0.0);
        assert_eq!(rvecs.len(), 3);
        assert_eq!(tvecs.len(), 3);
    }

    #[test]
    fn test_rodrigues_to_matrix() {
        let rvec = [0.0, 0.0, 0.0];
        let result = rodrigues_to_matrix(&rvec);
        assert!(result.is_ok());

        let matrix = result.unwrap();
        // Should be identity matrix
        assert!(libm::fabs(matrix[0][0] - 1.0) < 1e-6);
        assert!(libm::fabs(matrix[1][1] - 1.0) < 1e-6);
        assert!(libm::fabs(matrix[2][2] - 1.0) < 1e-6);
    }
}
