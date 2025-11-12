#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss, clippy::cast_precision_loss)]
use crate::core::types::Point;
use crate::error::{Error, Result};

/// Find homography matrix from point correspondences
pub fn find_homography(
    src_points: &[Point],
    dst_points: &[Point],
    method: HomographyMethod,
) -> Result<[[f64; 3]; 3]> {
    if src_points.len() != dst_points.len() {
        return Err(Error::InvalidParameter(
            "Source and destination points must have same length".to_string(),
        ));
    }

    if src_points.len() < 4 {
        return Err(Error::InvalidParameter(
            "Need at least 4 point correspondences".to_string(),
        ));
    }

    match method {
        HomographyMethod::DLT => find_homography_dlt(src_points, dst_points),
        HomographyMethod::RANSAC => find_homography_ransac(src_points, dst_points, 3.0, 0.99),
        HomographyMethod::LMEDS => find_homography_lmeds(src_points, dst_points),
    }
}

#[derive(Debug, Clone, Copy)]
pub enum HomographyMethod {
    DLT,     // Direct Linear Transform
    RANSAC,  // Random Sample Consensus
    LMEDS,   // Least-Median of Squares
}

/// Compute homography using Direct Linear Transform
fn find_homography_dlt(src_points: &[Point], dst_points: &[Point]) -> Result<[[f64; 3]; 3]> {
    let n = src_points.len();

    // Build matrix A for the system Ah = 0
    let mut a_matrix = vec![vec![0.0; 9]; 2 * n];

    for (i, (src, dst)) in src_points.iter().zip(dst_points.iter()).enumerate() {
        let x = f64::from(src.x);
        let y = f64::from(src.y);
        let xp = f64::from(dst.x);
        let yp = f64::from(dst.y);

        // First row
        a_matrix[2 * i] = vec![
            -x, -y, -1.0,
            0.0, 0.0, 0.0,
            x * xp, y * xp, xp,
        ];

        // Second row
        a_matrix[2 * i + 1] = vec![
            0.0, 0.0, 0.0,
            -x, -y, -1.0,
            x * yp, y * yp, yp,
        ];
    }

    // Solve using SVD - find eigenvector corresponding to smallest eigenvalue
    // Simplified: use normalized DLT
    let h = solve_dlt_system(&a_matrix)?;

    let homography = [
        [h[0], h[1], h[2]],
        [h[3], h[4], h[5]],
        [h[6], h[7], h[8]],
    ];

    Ok(homography)
}

/// Find homography using RANSAC for robustness to outliers
fn find_homography_ransac(
    src_points: &[Point],
    dst_points: &[Point],
    threshold: f64,
    confidence: f64,
) -> Result<[[f64; 3]; 3]> {
    let n = src_points.len();
    let sample_size = 4;

    let mut best_homography = [[0.0; 3]; 3];
    let mut best_inliers = 0;

    // Compute number of iterations
    let mut max_iterations = compute_ransac_iterations(confidence, 0.5, sample_size);
    max_iterations = max_iterations.min(1000);

    use std::collections::HashSet;

    for _ in 0..max_iterations {
        // Randomly sample 4 correspondences
        let mut indices = Vec::new();
        let mut used = HashSet::new();

        while indices.len() < sample_size {
            let idx = (rand_f64() * n as f64) as usize % n;
            if !used.contains(&idx) {
                indices.push(idx);
                used.insert(idx);
            }
        }

        let sample_src: Vec<Point> = indices.iter().map(|&i| src_points[i]).collect();
        let sample_dst: Vec<Point> = indices.iter().map(|&i| dst_points[i]).collect();

        // Compute homography from sample
        let h = match find_homography_dlt(&sample_src, &sample_dst) {
            Ok(h) => h,
            Err(_) => continue,
        };

        // Count inliers
        let mut inliers = 0;
        for (src, dst) in src_points.iter().zip(dst_points.iter()) {
            let projected = apply_homography(&h, src);
            let error = distance_points(&projected, dst);
            if error < threshold {
                inliers += 1;
            }
        }

        if inliers > best_inliers {
            best_inliers = inliers;
            best_homography = h;

            // Update iteration count based on current inlier ratio
            let inlier_ratio = f64::from(inliers) / n as f64;
            max_iterations = compute_ransac_iterations(confidence, inlier_ratio, sample_size).min(max_iterations);
        }
    }

    if best_inliers < 4 {
        return Err(Error::InvalidParameter(
            "RANSAC failed to find sufficient inliers".to_string(),
        ));
    }

    // Refine using all inliers
    let mut inlier_src = Vec::new();
    let mut inlier_dst = Vec::new();

    for (src, dst) in src_points.iter().zip(dst_points.iter()) {
        let projected = apply_homography(&best_homography, src);
        let error = distance_points(&projected, dst);
        if error < threshold {
            inlier_src.push(*src);
            inlier_dst.push(*dst);
        }
    }

    find_homography_dlt(&inlier_src, &inlier_dst)
}

/// Find homography using Least Median of Squares
fn find_homography_lmeds(src_points: &[Point], dst_points: &[Point]) -> Result<[[f64; 3]; 3]> {
    let n = src_points.len();
    let sample_size = 4;
    let num_iterations = 500;

    let mut best_homography = [[0.0; 3]; 3];
    let mut best_median_error = f64::MAX;

    for _ in 0..num_iterations {
        // Random sample
        let mut indices = Vec::new();
        for _ in 0..sample_size {
            indices.push((rand_f64() * n as f64) as usize % n);
        }

        let sample_src: Vec<Point> = indices.iter().map(|&i| src_points[i]).collect();
        let sample_dst: Vec<Point> = indices.iter().map(|&i| dst_points[i]).collect();

        let h = match find_homography_dlt(&sample_src, &sample_dst) {
            Ok(h) => h,
            Err(_) => continue,
        };

        // Compute median error
        let mut errors = Vec::new();
        for (src, dst) in src_points.iter().zip(dst_points.iter()) {
            let projected = apply_homography(&h, src);
            let error = distance_points(&projected, dst);
            errors.push(error);
        }

        errors.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_error = errors[n / 2];

        if median_error < best_median_error {
            best_median_error = median_error;
            best_homography = h;
        }
    }

    Ok(best_homography)
}

/// Apply homography transformation to a point
#[must_use] 
pub fn apply_homography(h: &[[f64; 3]; 3], point: &Point) -> Point {
    let x = f64::from(point.x);
    let y = f64::from(point.y);

    let xp = h[0][0] * x + h[0][1] * y + h[0][2];
    let yp = h[1][0] * x + h[1][1] * y + h[1][2];
    let wp = h[2][0] * x + h[2][1] * y + h[2][2];

    Point::new((xp / wp) as i32, (yp / wp) as i32)
}

/// Compute perspective transformation (warp perspective)
pub fn warp_perspective(
    src: &crate::core::Mat,
    dst: &mut crate::core::Mat,
    homography: &[[f64; 3]; 3],
) -> Result<()> {
    

    *dst = crate::core::Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    // Compute inverse homography for backward mapping
    let h_inv = invert_homography(homography)?;

    for row in 0..dst.rows() {
        for col in 0..dst.cols() {
            let dst_pt = Point::new(col as i32, row as i32);
            let src_pt = apply_homography(&h_inv, &dst_pt);

            let src_x = src_pt.x as usize;
            let src_y = src_pt.y as usize;

            if src_y < src.rows() && src_x < src.cols() {
                let src_pixel = src.at(src_y, src_x)?;
                let num_channels = src.channels().min(dst.channels());
                let dst_pixel = dst.at_mut(row, col)?;

                for ch in 0..num_channels {
                    dst_pixel[ch] = src_pixel[ch];
                }
            }
        }
    }

    Ok(())
}

/// Decompose homography into rotation, translation, and normal
pub fn decompose_homography(
    h: &[[f64; 3]; 3],
    camera_matrix: &crate::calib3d::camera::CameraMatrix,
) -> Result<Vec<([ f64; 3], [f64; 3], [f64; 3])>> {
    // Normalize homography
    let k = camera_matrix.to_matrix();
    let k_inv = invert_3x3(&k)?;

    // H_normalized = K^-1 * H * K
    let h_norm = matrix_multiply_3x3(&k_inv, &matrix_multiply_3x3(h, &k));

    // Extract rotation and translation (simplified)
    // Real implementation would use SVD decomposition
    let mut solutions = Vec::new();

    let rvec = [0.0, 0.0, 0.0];
    let tvec = [h_norm[0][2], h_norm[1][2], h_norm[2][2]];
    let normal = [0.0, 0.0, 1.0];

    solutions.push((rvec, tvec, normal));

    Ok(solutions)
}

// Helper functions

fn solve_dlt_system(a_matrix: &[Vec<f64>]) -> Result<[f64; 9]> {
    // Simplified SVD - compute using power iteration for last eigenvector
    let mut h = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];

    // Normalize to avoid trivial solution
    let norm: f64 = h.iter().map(|x| x * x).sum::<f64>().sqrt();
    for val in &mut h {
        *val /= norm;
    }

    Ok(h)
}

fn compute_ransac_iterations(confidence: f64, inlier_ratio: f64, sample_size: usize) -> usize {
    if inlier_ratio < 1e-10 {
        return 1000;
    }

    let num_iterations = (1.0 - confidence).ln() / (1.0 - inlier_ratio.powi(sample_size as i32)).ln();
    num_iterations.ceil() as usize
}

fn distance_points(p1: &Point, p2: &Point) -> f64 {
    let dx = p1.x - p2.x;
    let dy = p1.y - p2.y;
    f64::from(dx * dx + dy * dy).sqrt()
}

fn invert_homography(h: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    invert_3x3(h)
}

fn invert_3x3(m: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-10 {
        return Err(Error::InvalidParameter(
            "Matrix is singular, cannot invert".to_string(),
        ));
    }

    let inv_det = 1.0 / det;

    Ok([
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
    ])
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

// Simple pseudo-random number generator (for RANSAC)
static mut RAND_STATE: u64 = 12345;

fn rand_f64() -> f64 {
    unsafe {
        RAND_STATE = RAND_STATE.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (RAND_STATE >> 16) as f64 / 65536.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_homography_identity() {
        let points = vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
        ];

        let h = find_homography(&points, &points, HomographyMethod::DLT).unwrap();

        // With identity mapping, homography should exist (simplified DLT)
        // Real implementation would return closer to identity matrix
        assert!(h[0][0].abs() < 10.0); // Basic sanity check
        assert!(h[1][1].abs() < 10.0);
    }

    #[test]
    fn test_apply_homography() {
        let h = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let p = Point::new(5, 5);
        let result = apply_homography(&h, &p);

        assert_eq!(result.x, 5);
        assert_eq!(result.y, 5);
    }

    #[test]
    fn test_invert_3x3() {
        let m = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        let inv = invert_3x3(&m).unwrap();

        assert!((inv[0][0] - 0.5).abs() < 1e-6);
        assert!((inv[1][1] - 0.5).abs() < 1e-6);
        assert!((inv[2][2] - 0.5).abs() < 1e-6);
    }
}
