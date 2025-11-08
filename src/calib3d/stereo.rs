use crate::core::types::{Point, Point3f};
use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};
use crate::calib3d::camera::{CameraMatrix, DistortionCoefficients};

/// Stereo camera parameters
#[derive(Debug, Clone)]
pub struct StereoParameters {
    pub camera_matrix_left: CameraMatrix,
    pub camera_matrix_right: CameraMatrix,
    pub dist_coeffs_left: DistortionCoefficients,
    pub dist_coeffs_right: DistortionCoefficients,
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
    pub essential_matrix: [[f64; 3]; 3],
    pub fundamental_matrix: [[f64; 3]; 3],
}

/// Calibrate stereo camera pair
pub fn stereo_calibrate(
    object_points: &[Vec<Point3f>],
    image_points_left: &[Vec<Point>],
    image_points_right: &[Vec<Point>],
    image_size: (usize, usize),
) -> Result<StereoParameters> {
    if object_points.len() != image_points_left.len() ||
       object_points.len() != image_points_right.len() {
        return Err(Error::InvalidParameter(
            "Point arrays must have same length".to_string(),
        ));
    }

    // Calibrate individual cameras first
    use crate::calib3d::camera::calibrate_camera;

    let (camera_left, dist_left, _) = calibrate_camera(
        object_points,
        image_points_left,
        image_size,
    )?;

    let (camera_right, dist_right, _) = calibrate_camera(
        object_points,
        image_points_right,
        image_size,
    )?;

    // Estimate relative pose between cameras
    let (rotation, translation) = estimate_stereo_transform(
        object_points,
        image_points_left,
        image_points_right,
        &camera_left,
        &camera_right,
    )?;

    // Compute essential and fundamental matrices
    let essential = compute_essential_matrix(&rotation, &translation);
    let fundamental = compute_fundamental_matrix(&essential, &camera_left, &camera_right);

    Ok(StereoParameters {
        camera_matrix_left: camera_left,
        camera_matrix_right: camera_right,
        dist_coeffs_left: dist_left,
        dist_coeffs_right: dist_right,
        rotation,
        translation,
        essential_matrix: essential,
        fundamental_matrix: fundamental,
    })
}

fn estimate_stereo_transform(
    _object_points: &[Vec<Point3f>],
    _image_points_left: &[Vec<Point>],
    _image_points_right: &[Vec<Point>],
    _camera_left: &CameraMatrix,
    _camera_right: &CameraMatrix,
) -> Result<([[f64; 3]; 3], [f64; 3])> {
    // Simplified - real implementation would use bundle adjustment
    let rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let translation = [0.1, 0.0, 0.0]; // Baseline
    Ok((rotation, translation))
}

fn compute_essential_matrix(
    rotation: &[[f64; 3]; 3],
    translation: &[f64; 3],
) -> [[f64; 3]; 3] {
    // E = [t]_x * R
    // [t]_x is the skew-symmetric matrix of translation
    let tx = skew_symmetric(translation);

    matrix_multiply_3x3(&tx, rotation)
}

fn compute_fundamental_matrix(
    essential: &[[f64; 3]; 3],
    camera_left: &CameraMatrix,
    camera_right: &CameraMatrix,
) -> [[f64; 3]; 3] {
    // F = K_right^-T * E * K_left^-1
    let k_left = camera_left.to_matrix();
    let k_right = camera_right.to_matrix();

    let k_left_inv = matrix_inverse_3x3(&k_left);
    let k_right_inv_t = matrix_transpose_3x3(&matrix_inverse_3x3(&k_right));

    let temp = matrix_multiply_3x3(essential, &k_left_inv);
    matrix_multiply_3x3(&k_right_inv_t, &temp)
}

/// Compute disparity map from stereo images
pub fn compute_stereo_disparity(
    left: &Mat,
    right: &Mat,
    min_disparity: i32,
    max_disparity: i32,
    block_size: usize,
) -> Result<Mat> {
    if left.rows() != right.rows() || left.cols() != right.cols() {
        return Err(Error::InvalidDimensions(
            "Stereo images must have same size".to_string(),
        ));
    }

    if left.channels() != 1 || right.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Stereo matching requires grayscale images".to_string(),
        ));
    }

    let mut disparity = Mat::new(left.rows(), left.cols(), 1, MatDepth::F32)?;

    let half_block = block_size / 2;

    for row in half_block..left.rows() - half_block {
        for col in half_block..left.cols() - half_block {
            let mut best_disparity = 0;
            let mut best_cost = f32::MAX;

            // Search for best matching block in right image
            for d in min_disparity..max_disparity {
                let right_col = col as i32 - d;

                if right_col < half_block as i32 || right_col >= right.cols() as i32 - half_block as i32 {
                    continue;
                }

                // Compute SAD (Sum of Absolute Differences)
                let mut sad = 0.0f32;

                for dy in -(half_block as i32)..=(half_block as i32) {
                    for dx in -(half_block as i32)..=(half_block as i32) {
                        let y = (row as i32 + dy) as usize;
                        let x_left = (col as i32 + dx) as usize;
                        let x_right = (right_col + dx) as usize;

                        let val_left = left.at(y, x_left)?[0] as f32;
                        let val_right = right.at(y, x_right)?[0] as f32;

                        sad += (val_left - val_right).abs();
                    }
                }

                if sad < best_cost {
                    best_cost = sad;
                    best_disparity = d;
                }
            }

            let pixel = disparity.at_mut(row, col)?;
            pixel[0] = best_disparity as f32;
        }
    }

    Ok(disparity)
}

/// Triangulate 3D point from stereo correspondence
pub fn triangulate_point(
    point_left: Point,
    point_right: Point,
    stereo_params: &StereoParameters,
) -> Result<Point3f> {
    // Normalize image coordinates
    let x1 = (point_left.x as f64 - stereo_params.camera_matrix_left.cx) / stereo_params.camera_matrix_left.fx;
    let y1 = (point_left.y as f64 - stereo_params.camera_matrix_left.cy) / stereo_params.camera_matrix_left.fy;

    let x2 = (point_right.x as f64 - stereo_params.camera_matrix_right.cx) / stereo_params.camera_matrix_right.fx;
    let y2 = (point_right.y as f64 - stereo_params.camera_matrix_right.cy) / stereo_params.camera_matrix_right.fy;

    // Compute disparity
    let baseline = (stereo_params.translation[0] * stereo_params.translation[0] +
                    stereo_params.translation[1] * stereo_params.translation[1] +
                    stereo_params.translation[2] * stereo_params.translation[2]).sqrt();

    let disparity = x1 - x2;

    if disparity.abs() < 1e-6 {
        return Err(Error::InvalidParameter(
            "Disparity too small for triangulation".to_string(),
        ));
    }

    // Compute depth
    let depth = baseline * stereo_params.camera_matrix_left.fx / disparity;

    // Reconstruct 3D point
    let x = x1 * depth;
    let y = y1 * depth;
    let z = depth;

    Ok(Point3f::new(x as f32, y as f32, z as f32))
}

/// Rectify stereo images to align epipolar lines horizontally
pub fn stereo_rectify(
    stereo_params: &StereoParameters,
    image_size: (usize, usize),
) -> Result<(CameraMatrix, CameraMatrix, [[f64; 3]; 3], [[f64; 3]; 3])> {
    // Compute rectification transformations
    let (width, height) = image_size;

    // Rodrigues to get rotation vector
    let r_rect_left = compute_rectification_rotation(&stereo_params.rotation, true);
    let r_rect_right = compute_rectification_rotation(&stereo_params.rotation, false);

    // New camera matrices after rectification
    let mut new_camera_left = stereo_params.camera_matrix_left.clone();
    let mut new_camera_right = stereo_params.camera_matrix_right.clone();

    // Adjust principal points to align epipolar lines
    new_camera_left.cy = height as f64 / 2.0;
    new_camera_right.cy = height as f64 / 2.0;

    Ok((new_camera_left, new_camera_right, r_rect_left, r_rect_right))
}

fn compute_rectification_rotation(stereo_rotation: &[[f64; 3]; 3], _is_left: bool) -> [[f64; 3]; 3] {
    // Simplified rectification - real implementation would compute optimal rotation
    // to align epipolar lines with image rows
    *stereo_rotation
}

// Helper functions for matrix operations

fn skew_symmetric(v: &[f64; 3]) -> [[f64; 3]; 3] {
    [
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ]
}

fn matrix_multiply_3x3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut result = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    result
}

fn matrix_transpose_3x3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn matrix_inverse_3x3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    // Compute determinant
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-10 {
        return [[0.0; 3]; 3];
    }

    let inv_det = 1.0 / det;

    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
        ],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skew_symmetric() {
        let v = [1.0, 2.0, 3.0];
        let s = skew_symmetric(&v);

        assert_eq!(s[0][0], 0.0);
        assert_eq!(s[0][1], -3.0);
        assert_eq!(s[1][2], -1.0);
    }

    #[test]
    fn test_matrix_multiply() {
        let a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let result = matrix_multiply_3x3(&a, &b);

        assert_eq!(result[0][0], 2.0);
        assert_eq!(result[1][1], 2.0);
        assert_eq!(result[2][2], 2.0);
    }

    #[test]
    fn test_matrix_inverse() {
        let m = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let inv = matrix_inverse_3x3(&m);

        assert!((inv[0][0] - 0.5).abs() < 1e-6);
        assert!((inv[1][1] - 0.5).abs() < 1e-6);
        assert!((inv[2][2] - 0.5).abs() < 1e-6);
    }
}
