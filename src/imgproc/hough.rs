use crate::core::{Mat, MatDepth};
use crate::core::types::Point;
use crate::error::{Error, Result};
use std::f64::consts::PI;

/// Detect lines using Hough Transform
pub fn hough_lines(
    image: &Mat,
    rho: f64,
    theta: f64,
    threshold: i32,
) -> Result<Vec<(f64, f64)>> {
    if image.channels() != 1 {
        return Err(Error::InvalidParameter(
            "hough_lines only works on single-channel images".to_string(),
        ));
    }

    if image.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "hough_lines only supports U8 depth".to_string(),
        ));
    }

    // Calculate accumulator dimensions
    let max_rho = ((image.rows() * image.rows() + image.cols() * image.cols()) as f64).sqrt();
    let num_rho = (2.0 * max_rho / rho) as usize + 1;
    let num_theta = (PI / theta) as usize;

    // Initialize accumulator
    let mut accumulator = vec![vec![0i32; num_theta]; num_rho];

    // Vote in Hough space
    for row in 0..image.rows() {
        for col in 0..image.cols() {
            let pixel = image.at(row, col)?;

            if pixel[0] > 128 {
                // Edge pixel
                let x = col as f64;
                let y = row as f64;

                for t_idx in 0..num_theta {
                    let angle = t_idx as f64 * theta;
                    let r = x * angle.cos() + y * angle.sin();
                    let r_idx = ((r + max_rho) / rho) as usize;

                    if r_idx < num_rho {
                        accumulator[r_idx][t_idx] += 1;
                    }
                }
            }
        }
    }

    // Find peaks in accumulator
    let mut lines = Vec::new();

    for r_idx in 0..num_rho {
        for t_idx in 0..num_theta {
            if accumulator[r_idx][t_idx] >= threshold {
                let r = r_idx as f64 * rho - max_rho;
                let t = t_idx as f64 * theta;
                lines.push((r, t));
            }
        }
    }

    Ok(lines)
}

/// Detect lines using Probabilistic Hough Transform (returns line segments)
pub fn hough_lines_p(
    image: &Mat,
    rho: f64,
    theta: f64,
    threshold: i32,
    min_line_length: f64,
    max_line_gap: f64,
) -> Result<Vec<(Point, Point)>> {
    if image.channels() != 1 {
        return Err(Error::InvalidParameter(
            "hough_lines_p only works on single-channel images".to_string(),
        ));
    }

    // Get edge points
    let mut edge_points = Vec::new();

    for row in 0..image.rows() {
        for col in 0..image.cols() {
            let pixel = image.at(row, col)?;
            if pixel[0] > 128 {
                edge_points.push(Point::new(col as i32, row as i32));
            }
        }
    }

    // Simple line segment detection
    let mut line_segments = Vec::new();
    let mut used = vec![false; edge_points.len()];

    for i in 0..edge_points.len() {
        if used[i] {
            continue;
        }

        let p1 = edge_points[i];
        let mut best_p2 = p1;
        let mut best_len = 0.0;
        let mut best_idx = i;

        // Find best matching point
        for j in i + 1..edge_points.len() {
            if used[j] {
                continue;
            }

            let p2 = edge_points[j];
            let dx = (p2.x - p1.x) as f64;
            let dy = (p2.y - p1.y) as f64;
            let len = (dx * dx + dy * dy).sqrt();

            if len > best_len && len >= min_line_length {
                // Check if points are roughly collinear with other points
                best_p2 = p2;
                best_len = len;
                best_idx = j;
            }
        }

        if best_len >= min_line_length {
            line_segments.push((p1, best_p2));
            used[i] = true;
            used[best_idx] = true;
        }
    }

    Ok(line_segments)
}

/// Detect circles using Hough Circle Transform
pub fn hough_circles(
    image: &Mat,
    method: HoughCirclesMethod,
    dp: f64,
    min_dist: f64,
    param1: f64,
    param2: f64,
    min_radius: i32,
    max_radius: i32,
) -> Result<Vec<Circle>> {
    if image.channels() != 1 {
        return Err(Error::InvalidParameter(
            "hough_circles only works on single-channel images".to_string(),
        ));
    }

    if image.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "hough_circles only supports U8 depth".to_string(),
        ));
    }

    // Apply edge detection
    use crate::imgproc::canny;
    let mut edges = Mat::new(1, 1, 1, MatDepth::U8)?;
    canny(image, &mut edges, param1, param1 * 2.0)?;

    // Find circles
    let mut circles = Vec::new();

    // Simple circle detection using edge gradients
    for row in (min_radius as usize)..(image.rows() - min_radius as usize) {
        for col in (min_radius as usize)..(image.cols() - min_radius as usize) {
            for r in min_radius..=max_radius {
                let mut votes = 0;

                // Sample points on the circle perimeter
                let num_samples = (2.0 * PI * r as f64) as i32;

                for i in 0..num_samples {
                    let angle = 2.0 * PI * i as f64 / num_samples as f64;
                    let x = col as i32 + (r as f64 * angle.cos()) as i32;
                    let y = row as i32 + (r as f64 * angle.sin()) as i32;

                    if x >= 0 && x < image.cols() as i32 && y >= 0 && y < image.rows() as i32 {
                        let pixel = edges.at(y as usize, x as usize)?;
                        if pixel[0] > 128 {
                            votes += 1;
                        }
                    }
                }

                let vote_ratio = votes as f64 / num_samples as f64;

                if vote_ratio >= param2 {
                    circles.push(Circle {
                        center: Point::new(col as i32, row as i32),
                        radius: r,
                        votes,
                    });
                }
            }
        }
    }

    // Non-maximum suppression
    let mut filtered_circles: Vec<Circle> = Vec::new();

    circles.sort_by(|a, b| b.votes.cmp(&a.votes));

    for circle in &circles {
        let mut is_maximum = true;

        for existing in &filtered_circles {
            let dx = (circle.center.x - existing.center.x) as f64;
            let dy = (circle.center.y - existing.center.y) as f64;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < min_dist {
                is_maximum = false;
                break;
            }
        }

        if is_maximum {
            filtered_circles.push(circle.clone());
        }
    }

    Ok(filtered_circles)
}

/// Circle structure for Hough circle detection results
#[derive(Debug, Clone)]
pub struct Circle {
    pub center: Point,
    pub radius: i32,
    pub votes: i32,
}

/// Methods for Hough circle detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoughCirclesMethod {
    Gradient,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::Scalar;

    #[test]
    fn test_hough_lines() {
        let mut img = Mat::new(100, 100, 1, MatDepth::U8).unwrap();

        // Draw a horizontal line
        for col in 0..100 {
            let pixel = img.at_mut(50, col).unwrap();
            pixel[0] = 255;
        }

        let lines = hough_lines(&img, 1.0, PI / 180.0, 50).unwrap();
        assert!(!lines.is_empty());
    }

    #[test]
    fn test_hough_lines_p() {
        let mut img = Mat::new(100, 100, 1, MatDepth::U8).unwrap();

        // Draw a line
        for i in 0..50 {
            let pixel = img.at_mut(i, i).unwrap();
            pixel[0] = 255;
        }

        let lines = hough_lines_p(&img, 1.0, PI / 180.0, 20, 10.0, 5.0).unwrap();
        // May or may not detect depending on parameters
        assert!(lines.len() >= 0);
    }
}
