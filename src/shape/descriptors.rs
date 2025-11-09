use crate::core::types::{Point, Rect};

/// Compute perimeter of a contour
pub fn arc_length(contour: &[Point], closed: bool) -> f64 {
    if contour.len() < 2 {
        return 0.0;
    }

    let mut length = 0.0;

    let end = if closed {
        contour.len()
    } else {
        contour.len() - 1
    };

    for i in 0..end {
        let p1 = &contour[i];
        let p2 = &contour[(i + 1) % contour.len()];

        let dx = (p2.x - p1.x) as f64;
        let dy = (p2.y - p1.y) as f64;

        length += (dx * dx + dy * dy).sqrt();
    }

    length
}

/// Compute area of a contour
pub fn contour_area(contour: &[Point]) -> f64 {
    if contour.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;

    for i in 0..contour.len() {
        let j = (i + 1) % contour.len();
        area += (contour[i].x * contour[j].y) as f64;
        area -= (contour[j].x * contour[i].y) as f64;
    }

    area.abs() / 2.0
}

/// Compute circularity (4π·area / perimeter²)
pub fn circularity(contour: &[Point]) -> f64 {
    let area = contour_area(contour);
    let perimeter = arc_length(contour, true);

    if perimeter == 0.0 {
        return 0.0;
    }

    4.0 * std::f64::consts::PI * area / (perimeter * perimeter)
}

/// Compute convexity (convex hull area / contour area)
pub fn convexity(contour: &[Point]) -> f64 {
    let area = contour_area(contour);

    if area == 0.0 {
        return 0.0;
    }

    let hull = convex_hull(contour);
    let hull_area = contour_area(&hull);

    area / hull_area.max(1e-10)
}

/// Compute convex hull using Graham scan
pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    if points.len() < 3 {
        return points.to_vec();
    }

    let mut pts = points.to_vec();

    // Find lowest point (or leftmost if tie)
    let mut min_idx = 0;
    for (i, p) in pts.iter().enumerate() {
        if p.y < pts[min_idx].y || (p.y == pts[min_idx].y && p.x < pts[min_idx].x) {
            min_idx = i;
        }
    }

    pts.swap(0, min_idx);
    let anchor = pts[0];

    // Sort by polar angle
    pts[1..].sort_by(|a, b| {
        let angle_a = ((a.y - anchor.y) as f64).atan2((a.x - anchor.x) as f64);
        let angle_b = ((b.y - anchor.y) as f64).atan2((b.x - anchor.x) as f64);

        angle_a.partial_cmp(&angle_b).unwrap()
    });

    let mut hull = Vec::new();
    hull.push(pts[0]);
    hull.push(pts[1]);

    for i in 2..pts.len() {
        while hull.len() > 1 && ccw(&hull[hull.len() - 2], &hull[hull.len() - 1], &pts[i]) <= 0.0 {
            hull.pop();
        }
        hull.push(pts[i]);
    }

    hull
}

fn ccw(p1: &Point, p2: &Point, p3: &Point) -> f64 {
    ((p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)) as f64
}

/// Compute minimum enclosing circle (Welzl's algorithm)
pub fn min_enclosing_circle(points: &[Point]) -> (Point, f32) {
    if points.is_empty() {
        return (Point::new(0, 0), 0.0);
    }

    let mut pts = points.to_vec();
    let n = pts.len();
    min_circle_recursive(&mut pts, n, &mut Vec::new(), 0)
}

fn min_circle_recursive(
    points: &mut [Point],
    n: usize,
    boundary: &mut Vec<Point>,
    b: usize,
) -> (Point, f32) {
    if b == 3 {
        return circle_from_3_points(&boundary[0], &boundary[1], &boundary[2]);
    }

    if n == 0 {
        return match b {
            0 => (Point::new(0, 0), 0.0),
            1 => (boundary[0], 0.0),
            2 => circle_from_2_points(&boundary[0], &boundary[1]),
            _ => (Point::new(0, 0), 0.0),
        };
    }

    let idx = n - 1;
    let p = points[idx];

    let (center, radius) = min_circle_recursive(points, n - 1, boundary, b);

    if is_inside(&p, &center, radius) {
        return (center, radius);
    }

    boundary.push(p);
    let result = min_circle_recursive(points, n - 1, boundary, b + 1);
    boundary.pop();

    result
}

fn circle_from_2_points(p1: &Point, p2: &Point) -> (Point, f32) {
    let cx = (p1.x + p2.x) / 2;
    let cy = (p1.y + p2.y) / 2;
    let r = distance(p1, p2) / 2.0;

    (Point::new(cx, cy), r)
}

fn circle_from_3_points(p1: &Point, p2: &Point, p3: &Point) -> (Point, f32) {
    let ax = p1.x as f64;
    let ay = p1.y as f64;
    let bx = p2.x as f64;
    let by = p2.y as f64;
    let cx = p3.x as f64;
    let cy = p3.y as f64;

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));

    if d.abs() < 1e-10 {
        // Points are collinear
        return circle_from_2_points(p1, p2);
    }

    let ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d;
    let uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d;

    let center = Point::new(ux as i32, uy as i32);
    let radius = distance(p1, &center);

    (center, radius)
}

fn distance(p1: &Point, p2: &Point) -> f32 {
    let dx = (p2.x - p1.x) as f32;
    let dy = (p2.y - p1.y) as f32;
    (dx * dx + dy * dy).sqrt()
}

fn is_inside(p: &Point, center: &Point, radius: f32) -> bool {
    distance(p, center) <= radius * 1.001 // Small tolerance
}

/// Compute bounding rectangle
pub fn bounding_rect(points: &[Point]) -> Rect {
    if points.is_empty() {
        return Rect::new(0, 0, 0, 0);
    }

    let mut min_x = points[0].x;
    let mut max_x = points[0].x;
    let mut min_y = points[0].y;
    let mut max_y = points[0].y;

    for p in points {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }

    Rect::new(min_x, min_y, max_x - min_x, max_y - min_y)
}

/// Compute aspect ratio (width / height)
pub fn aspect_ratio(contour: &[Point]) -> f64 {
    let rect = bounding_rect(contour);

    if rect.height == 0 {
        return f64::INFINITY;
    }

    rect.width as f64 / rect.height as f64
}

/// Compute extent (contour area / bounding rect area)
pub fn extent(contour: &[Point]) -> f64 {
    let area = contour_area(contour);
    let rect = bounding_rect(contour);
    let rect_area = (rect.width * rect.height) as f64;

    if rect_area == 0.0 {
        return 0.0;
    }

    area / rect_area
}

/// Compute solidity (contour area / convex hull area)
pub fn solidity(contour: &[Point]) -> f64 {
    let area = contour_area(contour);
    let hull = convex_hull(contour);
    let hull_area = contour_area(&hull);

    if hull_area == 0.0 {
        return 0.0;
    }

    area / hull_area
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arc_length() {
        let contour = vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
        ];

        let length = arc_length(&contour, true);

        assert!((length - 40.0).abs() < 0.1);
    }

    #[test]
    fn test_contour_area() {
        let contour = vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
        ];

        let area = contour_area(&contour);

        assert!((area - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_circularity() {
        let contour = vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(10, 10),
            Point::new(0, 10),
        ];

        let circ = circularity(&contour);

        assert!(circ > 0.0 && circ <= 1.0);
    }

    #[test]
    fn test_convex_hull() {
        let points = vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(5, 5),
            Point::new(10, 10),
            Point::new(0, 10),
        ];

        let hull = convex_hull(&points);

        assert!(hull.len() >= 3);
        assert!(hull.len() <= points.len());
    }

    #[test]
    fn test_min_enclosing_circle() {
        let points = vec![
            Point::new(0, 0),
            Point::new(10, 0),
            Point::new(5, 10),
        ];

        let (center, radius) = min_enclosing_circle(&points);

        assert!(radius > 0.0);
        assert!(center.x >= 0 && center.x <= 10);
        assert!(center.y >= 0 && center.y <= 10);
    }

    #[test]
    fn test_bounding_rect() {
        let points = vec![
            Point::new(5, 5),
            Point::new(15, 10),
            Point::new(10, 20),
        ];

        let rect = bounding_rect(&points);

        assert_eq!(rect.x, 5);
        assert_eq!(rect.y, 5);
        assert_eq!(rect.width, 10);
        assert_eq!(rect.height, 15);
    }

    #[test]
    fn test_aspect_ratio() {
        let contour = vec![
            Point::new(0, 0),
            Point::new(20, 0),
            Point::new(20, 10),
            Point::new(0, 10),
        ];

        let ratio = aspect_ratio(&contour);

        assert!((ratio - 2.0).abs() < 0.1);
    }
}
