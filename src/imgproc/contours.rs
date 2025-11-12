use crate::core::{Mat, MatDepth};
use crate::core::types::Point;
use crate::error::{Error, Result};

/// Contour retrieval modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrievalMode {
    External,
    List,
    CComp,
    Tree,
}

/// Contour approximation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChainApproxMode {
    None,
    Simple,
    TC89L1,
    TC89KCOS,
}

/// Represents a contour as a vector of points
pub type Contour = Vec<Point>;

/// Find contours in a binary image
pub fn find_contours(
    image: &Mat,
    mode: RetrievalMode,
    method: ChainApproxMode,
) -> Result<Vec<Contour>> {
    if image.channels() != 1 {
        return Err(Error::InvalidParameter(
            "find_contours only works on single-channel images".to_string(),
        ));
    }

    if image.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "find_contours only supports U8 depth".to_string(),
        ));
    }

    let mut contours = Vec::new();
    let mut visited = vec![vec![false; image.cols()]; image.rows()];

    // Simple contour tracing algorithm
    for row in 1..image.rows() - 1 {
        for col in 1..image.cols() - 1 {
            let pixel = image.at(row, col)?;

            if pixel[0] > 128 && !visited[row][col] {
                // Found a potential contour start
                let contour = trace_contour(image, row, col, &mut visited)?;

                if !contour.is_empty() {
                    contours.push(contour);
                }
            }
        }
    }

    Ok(contours)
}

/// Trace a single contour starting from a point
fn trace_contour(
    image: &Mat,
    start_row: usize,
    start_col: usize,
    visited: &mut [Vec<bool>],
) -> Result<Contour> {
    let mut contour = Vec::new();
    let mut stack = vec![(start_row, start_col)];

    while let Some((row, col)) = stack.pop() {
        if visited[row][col] {
            continue;
        }

        let pixel = image.at(row, col)?;
        if pixel[0] <= 128 {
            continue;
        }

        visited[row][col] = true;
        contour.push(Point::new(col as i32, row as i32));

        // Check 8-connected neighbors
        for dy in -1..=1 {
            for dx in -1..=1 {
                if dy == 0 && dx == 0 {
                    continue;
                }

                let ny = row as i32 + dy;
                let nx = col as i32 + dx;

                if ny > 0 && ny < image.rows() as i32 - 1 && nx > 0 && nx < image.cols() as i32 - 1 {
                    let ny = ny as usize;
                    let nx = nx as usize;

                    if !visited[ny][nx] {
                        stack.push((ny, nx));
                    }
                }
            }
        }
    }

    Ok(contour)
}

/// Calculate contour area
#[must_use] 
pub fn contour_area(contour: &Contour) -> f64 {
    if contour.len() < 3 {
        return 0.0;
    }

    // Shoelace formula
    let mut area = 0.0;

    for i in 0..contour.len() {
        let p1 = contour[i];
        let p2 = contour[(i + 1) % contour.len()];

        area += f64::from(p1.x * p2.y - p2.x * p1.y);
    }

    (area / 2.0).abs()
}

/// Calculate arc length (perimeter) of contour
#[must_use] 
pub fn arc_length(contour: &Contour, closed: bool) -> f64 {
    if contour.is_empty() {
        return 0.0;
    }

    let mut length = 0.0;

    for i in 0..contour.len() - 1 {
        let p1 = contour[i];
        let p2 = contour[i + 1];

        let dx = f64::from(p2.x - p1.x);
        let dy = f64::from(p2.y - p1.y);

        length += (dx * dx + dy * dy).sqrt();
    }

    if closed && contour.len() > 1 {
        let p1 = contour[contour.len() - 1];
        let p2 = contour[0];

        let dx = f64::from(p2.x - p1.x);
        let dy = f64::from(p2.y - p1.y);

        length += (dx * dx + dy * dy).sqrt();
    }

    length
}

/// Approximate a contour with fewer points using Douglas-Peucker algorithm
#[must_use] 
pub fn approx_poly_dp(contour: &Contour, epsilon: f64, closed: bool) -> Contour {
    if contour.len() <= 2 {
        return contour.clone();
    }

    douglas_peucker(contour, epsilon, 0, contour.len() - 1)
}

fn douglas_peucker(points: &[Point], epsilon: f64, start: usize, end: usize) -> Vec<Point> {
    if end <= start + 1 {
        return vec![points[start], points[end]];
    }

    // Find the point with maximum distance
    let mut max_dist = 0.0;
    let mut max_idx = start;

    for i in start + 1..end {
        let dist = perpendicular_distance(points[i], points[start], points[end]);
        if dist > max_dist {
            max_dist = dist;
            max_idx = i;
        }
    }

    if max_dist > epsilon {
        // Recursively simplify
        let mut result1 = douglas_peucker(points, epsilon, start, max_idx);
        let result2 = douglas_peucker(points, epsilon, max_idx, end);

        result1.pop(); // Remove duplicate point
        result1.extend(result2);
        result1
    } else {
        vec![points[start], points[end]]
    }
}

fn perpendicular_distance(point: Point, line_start: Point, line_end: Point) -> f64 {
    let dx = f64::from(line_end.x - line_start.x);
    let dy = f64::from(line_end.y - line_start.y);

    let num = f64::from(((line_end.y - line_start.y) * point.x - (line_end.x - line_start.x) * point.y
        + line_end.x * line_start.y
        - line_end.y * line_start.x)
        .abs());

    let den = (dx * dx + dy * dy).sqrt();

    if den == 0.0 {
        0.0
    } else {
        num / den
    }
}

/// Calculate bounding rectangle for a contour
#[must_use] 
pub fn bounding_rect(contour: &Contour) -> crate::core::types::Rect {
    if contour.is_empty() {
        return crate::core::types::Rect::new(0, 0, 0, 0);
    }

    let mut min_x = contour[0].x;
    let mut max_x = contour[0].x;
    let mut min_y = contour[0].y;
    let mut max_y = contour[0].y;

    for point in contour {
        min_x = min_x.min(point.x);
        max_x = max_x.max(point.x);
        min_y = min_y.min(point.y);
        max_y = max_y.max(point.y);
    }

    crate::core::types::Rect::new(min_x, min_y, max_x - min_x, max_y - min_y)
}

/// Calculate moments of a contour
pub struct Moments {
    pub m00: f64,
    pub m10: f64,
    pub m01: f64,
    pub m20: f64,
    pub m11: f64,
    pub m02: f64,
}

impl Moments {
    #[must_use] 
    pub fn centroid(&self) -> (f64, f64) {
        if self.m00 == 0.0 {
            (0.0, 0.0)
        } else {
            (self.m10 / self.m00, self.m01 / self.m00)
        }
    }
}

#[must_use] 
pub fn moments(contour: &Contour) -> Moments {
    let mut m00 = 0.0;
    let mut m10 = 0.0;
    let mut m01 = 0.0;
    let mut m20 = 0.0;
    let mut m11 = 0.0;
    let mut m02 = 0.0;

    for point in contour {
        let x = f64::from(point.x);
        let y = f64::from(point.y);

        m00 += 1.0;
        m10 += x;
        m01 += y;
        m20 += x * x;
        m11 += x * y;
        m02 += y * y;
    }

    Moments {
        m00,
        m10,
        m01,
        m20,
        m11,
        m02,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contour_area() {
        let contour = vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
        ];

        let area = contour_area(&contour);
        assert!((area - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_arc_length() {
        let contour = vec![Point::new(0, 0), Point::new(10, 0), Point::new(10, 10)];

        let length = arc_length(&contour, false);
        assert!(length > 0.0);
    }

    #[test]
    fn test_bounding_rect() {
        let contour = vec![
            Point::new(5, 5),
            Point::new(15, 10),
            Point::new(20, 20),
        ];

        let rect = bounding_rect(&contour);
        assert_eq!(rect.x, 5);
        assert_eq!(rect.y, 5);
    }
}
