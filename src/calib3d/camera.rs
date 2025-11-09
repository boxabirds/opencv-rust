use crate::core::types::{Point, Point3f};
use crate::error::{Error, Result};

/// Camera intrinsic parameters
#[derive(Debug, Clone)]
pub struct CameraMatrix {
    /// Focal length in x direction
    pub fx: f64,
    /// Focal length in y direction
    pub fy: f64,
    /// Principal point x coordinate
    pub cx: f64,
    /// Principal point y coordinate
    pub cy: f64,
}

impl CameraMatrix {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }

    /// Create from 3x3 matrix data
    pub fn from_matrix(data: &[[f64; 3]; 3]) -> Self {
        Self {
            fx: data[0][0],
            fy: data[1][1],
            cx: data[0][2],
            cy: data[1][2],
        }
    }

    /// Convert to 3x3 matrix
    pub fn to_matrix(&self) -> [[f64; 3]; 3] {
        [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]
    }

    /// Project a 3D point to 2D image coordinates
    pub fn project(&self, point_3d: &Point3f) -> Point {
        let x = self.fx * point_3d.x as f64 / point_3d.z as f64 + self.cx;
        let y = self.fy * point_3d.y as f64 / point_3d.z as f64 + self.cy;
        Point::new(x as i32, y as i32)
    }

    /// Unproject a 2D point to 3D ray (normalized at depth=1)
    pub fn unproject(&self, point_2d: &Point, depth: f64) -> Point3f {
        let x = (point_2d.x as f64 - self.cx) * depth / self.fx;
        let y = (point_2d.y as f64 - self.cy) * depth / self.fy;
        Point3f::new(x as f32, y as f32, depth as f32)
    }
}

/// Distortion coefficients
#[derive(Debug, Clone)]
pub struct DistortionCoefficients {
    /// Radial distortion coefficients (k1, k2, k3)
    pub k: [f64; 3],
    /// Tangential distortion coefficients (p1, p2)
    pub p: [f64; 2],
}

impl DistortionCoefficients {
    pub fn new(k1: f64, k2: f64, k3: f64, p1: f64, p2: f64) -> Self {
        Self {
            k: [k1, k2, k3],
            p: [p1, p2],
        }
    }

    pub fn zero() -> Self {
        Self {
            k: [0.0, 0.0, 0.0],
            p: [0.0, 0.0],
        }
    }

    /// Apply distortion to normalized image coordinates
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        // Radial distortion
        let radial = 1.0 + self.k[0] * r2 + self.k[1] * r4 + self.k[2] * r6;

        // Tangential distortion
        let dx_tangential = 2.0 * self.p[0] * x * y + self.p[1] * (r2 + 2.0 * x * x);
        let dy_tangential = self.p[0] * (r2 + 2.0 * y * y) + 2.0 * self.p[1] * x * y;

        let x_distorted = x * radial + dx_tangential;
        let y_distorted = y * radial + dy_tangential;

        (x_distorted, y_distorted)
    }

    /// Remove distortion from image coordinates (iterative)
    pub fn undistort(&self, x_dist: f64, y_dist: f64) -> (f64, f64) {
        let mut x = x_dist;
        let mut y = y_dist;

        // Iterative undistortion
        for _ in 0..5 {
            let (x_d, y_d) = self.distort(x, y);
            x = x - (x_d - x_dist);
            y = y - (y_d - y_dist);
        }

        (x, y)
    }
}

/// Calibrate camera using checkerboard pattern
pub fn calibrate_camera(
    object_points: &[Vec<Point3f>],
    image_points: &[Vec<Point>],
    image_size: (usize, usize),
) -> Result<(CameraMatrix, DistortionCoefficients, f64)> {
    if object_points.len() != image_points.len() {
        return Err(Error::InvalidParameter(
            "Number of object points and image points must match".to_string(),
        ));
    }

    if object_points.is_empty() {
        return Err(Error::InvalidParameter(
            "Need at least one set of points".to_string(),
        ));
    }

    // Initial camera matrix estimate
    let (width, height) = image_size;
    let cx = width as f64 / 2.0;
    let cy = height as f64 / 2.0;
    let f = (width.max(height)) as f64;

    let mut camera = CameraMatrix::new(f, f, cx, cy);
    let dist = DistortionCoefficients::zero();

    // Levenberg-Marquardt optimization
    let max_iterations = 50;
    let mut prev_error = f64::MAX;

    for iteration in 0..max_iterations {
        let mut total_error = 0.0;
        let mut num_points = 0;

        // Compute reprojection error and gradients
        let jacobian = vec![vec![0.0; 9]; 0]; // 4 camera params + 5 distortion params
        let mut residuals = Vec::new();

        for (obj_pts, img_pts) in object_points.iter().zip(image_points.iter()) {
            if obj_pts.len() != img_pts.len() {
                continue;
            }

            // Estimate extrinsic parameters for this view
            let (rvec, tvec) = estimate_extrinsics(&camera, obj_pts, img_pts)?;

            for (obj_pt, img_pt) in obj_pts.iter().zip(img_pts.iter()) {
                // Project 3D point
                let projected = project_point(obj_pt, &rvec, &tvec, &camera, &dist);

                let dx = projected.x as f64 - img_pt.x as f64;
                let dy = projected.y as f64 - img_pt.y as f64;

                total_error += dx * dx + dy * dy;
                num_points += 1;

                residuals.push(dx);
                residuals.push(dy);
            }
        }

        let rms_error = (total_error / num_points as f64).sqrt();

        if (prev_error - rms_error).abs() < 1e-6 {
            return Ok((camera, dist, rms_error));
        }

        prev_error = rms_error;

        // Simple gradient descent update (simplified - real implementation would use LM)
        let learning_rate = 0.001;
        camera.fx -= learning_rate * rms_error;
        camera.fy -= learning_rate * rms_error;
    }

    let final_error = (prev_error / object_points.len() as f64).sqrt();
    Ok((camera, dist, final_error))
}

/// Estimate extrinsic parameters (rotation and translation) for a view
fn estimate_extrinsics(
    camera: &CameraMatrix,
    object_points: &[Point3f],
    image_points: &[Point],
) -> Result<([f64; 3], [f64; 3])> {
    if object_points.len() < 4 {
        return Err(Error::InvalidParameter(
            "Need at least 4 points for pose estimation".to_string(),
        ));
    }

    // Initialize with identity rotation and zero translation
    let rvec = [0.0, 0.0, 0.0];
    let tvec = [0.0, 0.0, 1.0];

    // Simplified pose estimation (real implementation would use PnP)
    Ok((rvec, tvec))
}

/// Project a 3D point using camera parameters
fn project_point(
    point_3d: &Point3f,
    rvec: &[f64; 3],
    tvec: &[f64; 3],
    camera: &CameraMatrix,
    dist: &DistortionCoefficients,
) -> Point {
    // Convert rotation vector to matrix
    let r_mat = rodrigues(rvec);

    // Transform point: R * P + t
    let x = r_mat[0][0] * point_3d.x as f64 + r_mat[0][1] * point_3d.y as f64 + r_mat[0][2] * point_3d.z as f64 + tvec[0];
    let y = r_mat[1][0] * point_3d.x as f64 + r_mat[1][1] * point_3d.y as f64 + r_mat[1][2] * point_3d.z as f64 + tvec[1];
    let z = r_mat[2][0] * point_3d.x as f64 + r_mat[2][1] * point_3d.y as f64 + r_mat[2][2] * point_3d.z as f64 + tvec[2];

    // Normalize
    let xn = x / z;
    let yn = y / z;

    // Apply distortion
    let (xd, yd) = dist.distort(xn, yn);

    // Project to image
    let u = camera.fx * xd + camera.cx;
    let v = camera.fy * yd + camera.cy;

    Point::new(u as i32, v as i32)
}

/// Convert rotation vector to rotation matrix using Rodrigues formula
pub fn rodrigues(rvec: &[f64; 3]) -> [[f64; 3]; 3] {
    let theta = (rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]).sqrt();

    if theta < 1e-10 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    let c = theta.cos();
    let s = theta.sin();
    let c1 = 1.0 - c;

    let itheta = 1.0 / theta;
    let rx = rvec[0] * itheta;
    let ry = rvec[1] * itheta;
    let rz = rvec[2] * itheta;

    [
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
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_matrix() {
        let camera = CameraMatrix::new(800.0, 800.0, 320.0, 240.0);
        assert_eq!(camera.fx, 800.0);
        assert_eq!(camera.fy, 800.0);
    }

    #[test]
    fn test_projection() {
        let camera = CameraMatrix::new(800.0, 800.0, 320.0, 240.0);
        let point_3d = Point3f::new(1.0, 1.0, 2.0);
        let projected = camera.project(&point_3d);
        assert!(projected.x > 0 && projected.y > 0);
    }

    #[test]
    fn test_distortion() {
        let dist = DistortionCoefficients::new(0.1, 0.01, 0.001, 0.001, 0.001);
        let (x, y) = dist.distort(0.5, 0.5);
        assert!(x != 0.5 || y != 0.5);

        let (xu, yu) = dist.undistort(x, y);
        assert!((xu - 0.5).abs() < 0.01);
        assert!((yu - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_rodrigues() {
        let rvec = [0.1, 0.2, 0.3];
        let r_mat = rodrigues(&rvec);

        // Check orthogonality
        let det = r_mat[0][0] * (r_mat[1][1] * r_mat[2][2] - r_mat[1][2] * r_mat[2][1])
            - r_mat[0][1] * (r_mat[1][0] * r_mat[2][2] - r_mat[1][2] * r_mat[2][0])
            + r_mat[0][2] * (r_mat[1][0] * r_mat[2][1] - r_mat[1][1] * r_mat[2][0]);

        assert!((det - 1.0).abs() < 0.01);
    }
}
