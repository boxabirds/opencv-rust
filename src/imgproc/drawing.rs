use crate::core::{Mat, MatDepth};
use crate::core::types::{Point, Rect, Scalar};
use crate::error::{Error, Result};

/// Draw a line on an image
pub fn line(
    img: &mut Mat,
    pt1: Point,
    pt2: Point,
    color: Scalar,
    thickness: i32,
) -> Result<()> {
    if img.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "Drawing only supports U8 depth".to_string(),
        ));
    }

    // Bresenham's line algorithm
    let mut x0 = pt1.x;
    let mut y0 = pt1.y;
    let x1 = pt2.x;
    let y1 = pt2.y;

    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        // Draw pixel (with thickness support via circle)
        if thickness > 1 {
            circle(img, Point::new(x0, y0), thickness / 2, color)?;
        } else if x0 >= 0 && x0 < img.cols() as i32 && y0 >= 0 && y0 < img.rows() as i32 {
            let num_channels = img.channels();
            let pixel = img.at_mut(y0 as usize, x0 as usize)?;
            for ch in 0..num_channels.min(4) {
                pixel[ch] = color.val[ch] as u8;
            }
        }

        if x0 == x1 && y0 == y1 {
            break;
        }

        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }

    Ok(())
}

/// Draw a rectangle on an image
pub fn rectangle(img: &mut Mat, rect: Rect, color: Scalar, thickness: i32) -> Result<()> {
    if thickness < 0 {
        // Filled rectangle
        let num_channels = img.channels();
        for y in rect.y..(rect.y + rect.height) {
            for x in rect.x..(rect.x + rect.width) {
                if y >= 0 && y < img.rows() as i32 && x >= 0 && x < img.cols() as i32 {
                    let pixel = img.at_mut(y as usize, x as usize)?;
                    for ch in 0..num_channels.min(4) {
                        pixel[ch] = color.val[ch] as u8;
                    }
                }
            }
        }
    } else {
        // Draw four lines
        let pt1 = Point::new(rect.x, rect.y);
        let pt2 = Point::new(rect.x + rect.width, rect.y);
        let pt3 = Point::new(rect.x + rect.width, rect.y + rect.height);
        let pt4 = Point::new(rect.x, rect.y + rect.height);

        line(img, pt1, pt2, color, thickness)?;
        line(img, pt2, pt3, color, thickness)?;
        line(img, pt3, pt4, color, thickness)?;
        line(img, pt4, pt1, color, thickness)?;
    }

    Ok(())
}

/// Draw a circle on an image
pub fn circle(img: &mut Mat, center: Point, radius: i32, color: Scalar) -> Result<()> {
    if img.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "Drawing only supports U8 depth".to_string(),
        ));
    }

    // Midpoint circle algorithm
    let mut x = radius;
    let mut y = 0;
    let mut err = 0;

    while x >= y {
        // Draw 8 octants
        let points = [
            (center.x + x, center.y + y),
            (center.x + y, center.y + x),
            (center.x - y, center.y + x),
            (center.x - x, center.y + y),
            (center.x - x, center.y - y),
            (center.x - y, center.y - x),
            (center.x + y, center.y - x),
            (center.x + x, center.y - y),
        ];

        let num_channels = img.channels();
        for &(px, py) in &points {
            if px >= 0 && px < img.cols() as i32 && py >= 0 && py < img.rows() as i32 {
                let pixel = img.at_mut(py as usize, px as usize)?;
                for ch in 0..num_channels.min(4) {
                    pixel[ch] = color.val[ch] as u8;
                }
            }
        }

        y += 1;
        if err <= 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err -= 2 * x + 1;
        }
    }

    Ok(())
}

/// Draw filled circle
pub fn circle_filled(img: &mut Mat, center: Point, radius: i32, color: Scalar) -> Result<()> {
    let r_squared = radius * radius;
    let num_channels = img.channels();

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx * dx + dy * dy <= r_squared {
                let x = center.x + dx;
                let y = center.y + dy;

                if x >= 0 && x < img.cols() as i32 && y >= 0 && y < img.rows() as i32 {
                    let pixel = img.at_mut(y as usize, x as usize)?;
                    for ch in 0..num_channels.min(4) {
                        pixel[ch] = color.val[ch] as u8;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Draw an ellipse on an image
pub fn ellipse(
    img: &mut Mat,
    center: Point,
    axes: (i32, i32),
    angle: f64,
    start_angle: f64,
    end_angle: f64,
    color: Scalar,
) -> Result<()> {
    if img.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "Drawing only supports U8 depth".to_string(),
        ));
    }

    let (a, b) = axes;
    let angle_rad = angle * std::f64::consts::PI / 180.0;
    let cos_angle = angle_rad.cos();
    let sin_angle = angle_rad.sin();

    // Sample points along the ellipse
    let num_points = 100;
    let angle_range = end_angle - start_angle;

    for i in 0..num_points {
        let t = start_angle + (f64::from(i) / f64::from(num_points)) * angle_range;
        let t_rad = t * std::f64::consts::PI / 180.0;

        // Parametric ellipse equation
        let x_local = f64::from(a) * t_rad.cos();
        let y_local = f64::from(b) * t_rad.sin();

        // Rotate
        let x_rot = x_local * cos_angle - y_local * sin_angle;
        let y_rot = x_local * sin_angle + y_local * cos_angle;

        // Translate
        let x = center.x + x_rot as i32;
        let y = center.y + y_rot as i32;

        let num_channels = img.channels();
        if x >= 0 && x < img.cols() as i32 && y >= 0 && y < img.rows() as i32 {
            let pixel = img.at_mut(y as usize, x as usize)?;
            for ch in 0..num_channels.min(4) {
                pixel[ch] = color.val[ch] as u8;
            }
        }
    }

    Ok(())
}

/// Draw a polyline
pub fn polylines(
    img: &mut Mat,
    pts: &[Point],
    is_closed: bool,
    color: Scalar,
    thickness: i32,
) -> Result<()> {
    if pts.len() < 2 {
        return Ok(());
    }

    for i in 0..pts.len() - 1 {
        line(img, pts[i], pts[i + 1], color, thickness)?;
    }

    if is_closed && pts.len() > 2 {
        line(img, pts[pts.len() - 1], pts[0], color, thickness)?;
    }

    Ok(())
}

/// Fill a polygon
pub fn fill_poly(img: &mut Mat, pts: &[Point], color: Scalar) -> Result<()> {
    if pts.len() < 3 {
        return Ok(());
    }

    // Find bounding box
    let mut min_x = pts[0].x;
    let mut max_x = pts[0].x;
    let mut min_y = pts[0].y;
    let mut max_y = pts[0].y;

    for pt in pts {
        min_x = min_x.min(pt.x);
        max_x = max_x.max(pt.x);
        min_y = min_y.min(pt.y);
        max_y = max_y.max(pt.y);
    }

    // Scan-line fill algorithm
    let num_channels = img.channels();
    for y in min_y..=max_y {
        let mut intersections = Vec::new();

        for i in 0..pts.len() {
            let p1 = pts[i];
            let p2 = pts[(i + 1) % pts.len()];

            if (p1.y <= y && p2.y > y) || (p2.y <= y && p1.y > y) {
                let x = p1.x + (f64::from(y - p1.y) / f64::from(p2.y - p1.y) * f64::from(p2.x - p1.x)) as i32;
                intersections.push(x);
            }
        }

        intersections.sort_unstable();

        for i in (0..intersections.len()).step_by(2) {
            if i + 1 < intersections.len() {
                for x in intersections[i]..=intersections[i + 1] {
                    if x >= 0 && x < img.cols() as i32 && y >= 0 && y < img.rows() as i32 {
                        let pixel = img.at_mut(y as usize, x as usize)?;
                        for ch in 0..num_channels.min(4) {
                            pixel[ch] = color.val[ch] as u8;
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Put text on an image (very basic, just draws rectangles for characters)
pub fn put_text(
    img: &mut Mat,
    text: &str,
    org: Point,
    font_scale: f64,
    color: Scalar,
) -> Result<()> {
    // Very simple text rendering - just draw rectangles to represent characters
    let char_width = (8.0 * font_scale) as i32;
    let char_height = (12.0 * font_scale) as i32;

    for (i, _ch) in text.chars().enumerate() {
        let x = org.x + i as i32 * char_width;
        let y = org.y;

        let rect = Rect::new(x, y, char_width - 2, char_height);
        rectangle(img, rect, color, 1)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line() {
        let mut img = Mat::new(100, 100, 3, MatDepth::U8).unwrap();
        line(&mut img, Point::new(10, 10), Point::new(90, 90), Scalar::from_rgb(255, 0, 0), 1).unwrap();
    }

    #[test]
    fn test_rectangle() {
        let mut img = Mat::new(100, 100, 3, MatDepth::U8).unwrap();
        rectangle(&mut img, Rect::new(10, 10, 50, 50), Scalar::from_rgb(0, 255, 0), 2).unwrap();
    }

    #[test]
    fn test_circle() {
        let mut img = Mat::new(100, 100, 3, MatDepth::U8).unwrap();
        circle(&mut img, Point::new(50, 50), 30, Scalar::from_rgb(0, 0, 255)).unwrap();
    }
}
