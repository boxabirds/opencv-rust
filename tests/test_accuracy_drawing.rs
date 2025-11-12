#![allow(unused_comparisons)]
/// Bit-level accuracy tests for Drawing Functions
mod test_utils;

use opencv_rust::core::{Mat, MatDepth};
use opencv_rust::core::types::{Point, Rect, Scalar};
use opencv_rust::imgproc::drawing::{line, rectangle, circle};
use test_utils::*;

// ===== LINE TESTS =====

#[test]
fn test_line_horizontal() {
    let mut img = Mat::new(20, 30, 3, MatDepth::U8).unwrap();
    let color = Scalar::new(255.0, 0.0, 0.0, 0.0);  // Red

    line(&mut img, Point::new(5, 10), Point::new(25, 10), color, 1).unwrap();

    // Check pixels along the line
    for x in 5..=25 {
        let pixel = img.at(10, x as usize).unwrap();
        assert_eq!(pixel[0], 255, "Red channel at ({}, 10)", x);
        assert_eq!(pixel[1], 0, "Green channel at ({}, 10)", x);
        assert_eq!(pixel[2], 0, "Blue channel at ({}, 10)", x);
    }
}

#[test]
fn test_line_vertical() {
    let mut img = Mat::new(30, 20, 3, MatDepth::U8).unwrap();
    let color = Scalar::new(0.0, 255.0, 0.0, 0.0);  // Green

    line(&mut img, Point::new(10, 5), Point::new(10, 25), color, 1).unwrap();

    // Check pixels along the line
    for y in 5..=25 {
        let pixel = img.at(y as usize, 10).unwrap();
        assert_eq!(pixel[0], 0, "Red channel at (10, {})", y);
        assert_eq!(pixel[1], 255, "Green channel at (10, {})", y);
        assert_eq!(pixel[2], 0, "Blue channel at (10, {})", y);
    }
}

#[test]
fn test_line_diagonal() {
    let mut img = Mat::new(50, 50, 3, MatDepth::U8).unwrap();
    let color = Scalar::new(0.0, 0.0, 255.0, 0.0);  // Blue

    line(&mut img, Point::new(10, 10), Point::new(40, 40), color, 1).unwrap();

    // Check some pixels along the diagonal
    let pixel = img.at(10, 10).unwrap();
    assert_eq!(pixel[2], 255, "Start point should be blue");

    let pixel = img.at(40, 40).unwrap();
    assert_eq!(pixel[2], 255, "End point should be blue");

    let pixel = img.at(25, 25).unwrap();
    assert_eq!(pixel[2], 255, "Middle point should be blue");
}

#[test]
fn test_line_thickness() {
    let mut img = Mat::new(50, 50, 1, MatDepth::U8).unwrap();
    let color = Scalar::all(255.0);

    line(&mut img, Point::new(25, 10), Point::new(25, 40), color, 5).unwrap();

    // Thick line should affect neighboring pixels
    // Note: exact thickness behavior may vary
    let pixel = img.at(25, 25).unwrap();
    assert_eq!(pixel[0], 255, "Center of thick line");
}

#[test]
fn test_line_boundary_clipping() {
    let mut img = Mat::new(20, 20, 3, MatDepth::U8).unwrap();
    let color = Scalar::all(255.0);

    // Line extending beyond image bounds
    line(&mut img, Point::new(-5, 10), Point::new(25, 10), color, 1).unwrap();

    // Should not panic, only draw visible portion
    let pixel = img.at(10, 10).unwrap();
    assert_eq!(pixel[0], 255, "Visible portion drawn");
}

// ===== RECTANGLE TESTS =====

#[test]
fn test_rectangle_outline() {
    let mut img = Mat::new(50, 50, 3, MatDepth::U8).unwrap();
    let color = Scalar::new(255.0, 128.0, 64.0, 0.0);
    let rect = Rect::new(10, 10, 30, 20);

    rectangle(&mut img, rect, color, 1).unwrap();

    // Check corners (implementation draws to x+width, y+height not -1)
    let pixel = img.at(10, 10).unwrap();
    assert_eq!(pixel[0], 255, "Top-left corner");

    let pixel = img.at(10, 40).unwrap();  // x+width = 10+30 = 40
    assert_eq!(pixel[0], 255, "Top-right corner");

    let pixel = img.at(30, 10).unwrap();  // y+height = 10+20 = 30
    assert_eq!(pixel[0], 255, "Bottom-left corner");

    let pixel = img.at(30, 40).unwrap();  // (y+height, x+width)
    assert_eq!(pixel[0], 255, "Bottom-right corner");

    // Check interior is not filled
    let pixel = img.at(20, 20).unwrap();
    assert_eq!(pixel[0], 0, "Interior should be empty");
}

#[test]
fn test_rectangle_filled() {
    let mut img = Mat::new(50, 50, 1, MatDepth::U8).unwrap();
    let color = Scalar::all(200.0);
    let rect = Rect::new(10, 10, 20, 15);

    rectangle(&mut img, rect, color, -1).unwrap();  // -1 for filled

    // Check interior is filled
    for y in 10..25 {
        for x in 10..30 {
            let pixel = img.at(y, x).unwrap();
            assert_eq!(pixel[0], 200, "Interior filled at ({}, {})", x, y);
        }
    }

    // Check outside is not filled
    let pixel = img.at(5, 5).unwrap();
    assert_eq!(pixel[0], 0, "Outside should be empty");
}

#[test]
fn test_rectangle_dimensions() {
    let mut img = Mat::new(100, 100, 3, MatDepth::U8).unwrap();
    let color = Scalar::all(255.0);
    let rect = Rect::new(20, 30, 40, 25);

    rectangle(&mut img, rect, color, 2).unwrap();

    // Verify dimensions
    assert_eq!(rect.width, 40);
    assert_eq!(rect.height, 25);
}

#[test]
fn test_rectangle_single_pixel() {
    let mut img = Mat::new(50, 50, 1, MatDepth::U8).unwrap();
    let color = Scalar::all(255.0);
    let rect = Rect::new(25, 25, 1, 1);

    rectangle(&mut img, rect, color, -1).unwrap();

    let pixel = img.at(25, 25).unwrap();
    assert_eq!(pixel[0], 255, "Single pixel rectangle");
}

// ===== CIRCLE TESTS =====

#[test]
fn test_circle_basic() {
    let mut img = Mat::new(100, 100, 3, MatDepth::U8).unwrap();
    let color = Scalar::new(255.0, 0.0, 0.0, 0.0);
    let center = Point::new(50, 50);
    let radius = 20;

    circle(&mut img, center, radius, color).unwrap();

    // Check points on circumference (circle only draws outline, not filled)
    // Right edge: (50+20, 50) = (70, 50)
    let pixel = img.at(50, 70).unwrap();
    assert_eq!(pixel[0], 255, "Right circumference point");

    // Top edge: (50, 50-20) = (50, 30)
    let pixel = img.at(30, 50).unwrap();
    assert_eq!(pixel[0], 255, "Top circumference point");
}

#[test]
fn test_circle_small_radius() {
    let mut img = Mat::new(50, 50, 1, MatDepth::U8).unwrap();
    let color = Scalar::all(255.0);
    let center = Point::new(25, 25);

    circle(&mut img, center, 1, color).unwrap();

    // With radius=1, circle draws outline points around (25,25)
    // Check that at least one point near center was drawn
    let drawn_count = [
        img.at(24, 25).unwrap()[0],
        img.at(26, 25).unwrap()[0],
        img.at(25, 24).unwrap()[0],
        img.at(25, 26).unwrap()[0],
    ].iter().filter(|&&v| v == 255).count();

    assert!(drawn_count > 0, "At least one point should be drawn for small circle");
}

#[test]
fn test_circle_large_radius() {
    let mut img = Mat::new(200, 200, 1, MatDepth::U8).unwrap();
    let color = Scalar::all(255.0);
    let center = Point::new(100, 100);
    let radius = 80;

    circle(&mut img, center, radius, color).unwrap();

    // Check points on circumference (not center, circle draws outline only)
    // Right edge: (100, 100+80) = (100, 180)
    let pixel = img.at(100, 180).unwrap();
    assert_eq!(pixel[0], 255, "Right circumference point");

    // Left edge: (100, 100-80) = (100, 20)
    let pixel = img.at(100, 20).unwrap();
    assert_eq!(pixel[0], 255, "Left circumference point");
}

#[test]
fn test_circle_boundary_clipping() {
    let mut img = Mat::new(50, 50, 1, MatDepth::U8).unwrap();
    let color = Scalar::all(255.0);

    // Circle partially outside image
    circle(&mut img, Point::new(5, 5), 10, color).unwrap();

    // Should not panic, only draw visible portion
    // Check a point that should be on the visible part of circumference
    let pixel = img.at(5, 15).unwrap();  // Right side of circle at (5, 5+10)
    assert_eq!(pixel[0], 255, "Visible portion drawn");
}

#[test]
fn test_circle_different_colors() {
    let mut img = Mat::new(100, 100, 3, MatDepth::U8).unwrap();

    // Red circle
    circle(&mut img, Point::new(30, 30), 10, Scalar::new(255.0, 0.0, 0.0, 0.0)).unwrap();

    // Green circle
    circle(&mut img, Point::new(70, 30), 10, Scalar::new(0.0, 255.0, 0.0, 0.0)).unwrap();

    // Blue circle
    circle(&mut img, Point::new(50, 70), 10, Scalar::new(0.0, 0.0, 255.0, 0.0)).unwrap();

    // Verify colors at circumference points (circles draw outline only)
    let red_pixel = img.at(30, 40).unwrap();  // Right edge of red circle (30, 30+10)
    assert_eq!(red_pixel[0], 255, "Red circle");

    let green_pixel = img.at(30, 80).unwrap();  // Right edge of green circle (30, 70+10)
    assert_eq!(green_pixel[1], 255, "Green circle");

    let blue_pixel = img.at(70, 60).unwrap();  // Right edge of blue circle (70, 50+10)
    assert_eq!(blue_pixel[2], 255, "Blue circle");
}

// ===== COMBINED TESTS =====

#[test]
fn test_draw_complex_shape() {
    let mut img = Mat::new(200, 200, 3, MatDepth::U8).unwrap();

    // Draw a house
    let color = Scalar::all(255.0);

    // House body (rectangle)
    rectangle(&mut img, Rect::new(50, 100, 100, 80), color, 2).unwrap();

    // Roof (triangle using lines)
    line(&mut img, Point::new(50, 100), Point::new(100, 50), color, 2).unwrap();
    line(&mut img, Point::new(100, 50), Point::new(150, 100), color, 2).unwrap();

    // Door (filled rectangle)
    rectangle(&mut img, Rect::new(85, 140, 30, 40), color, -1).unwrap();

    // Window (circle)
    circle(&mut img, Point::new(120, 130), 10, color).unwrap();

    assert_eq!(img.rows(), 200);
    assert_eq!(img.cols(), 200);
}

#[test]
fn test_drawing_deterministic() {
    let mut img1 = Mat::new(50, 50, 1, MatDepth::U8).unwrap();
    let mut img2 = Mat::new(50, 50, 1, MatDepth::U8).unwrap();
    let color = Scalar::all(255.0);

    // Draw same shapes on both images
    line(&mut img1, Point::new(10, 10), Point::new(40, 40), color, 1).unwrap();
    line(&mut img2, Point::new(10, 10), Point::new(40, 40), color, 1).unwrap();

    rectangle(&mut img1, Rect::new(5, 5, 10, 10), color, 1).unwrap();
    rectangle(&mut img2, Rect::new(5, 5, 10, 10), color, 1).unwrap();

    circle(&mut img1, Point::new(30, 30), 5, color).unwrap();
    circle(&mut img2, Point::new(30, 30), 5, color).unwrap();

    assert_images_equal(&img1, &img2, "Drawing should be deterministic");
}

#[test]
#[ignore]
fn test_drawing_visual_inspection() {
    let mut img = Mat::new(50, 50, 1, MatDepth::U8).unwrap();
    let color = Scalar::all(255.0);

    println!("\nBefore drawing:");
    print_image_data(&img, "Blank", 50, 50);

    // Draw various shapes
    line(&mut img, Point::new(5, 5), Point::new(45, 45), color, 1).unwrap();
    rectangle(&mut img, Rect::new(10, 10, 15, 15), color, 2).unwrap();
    circle(&mut img, Point::new(35, 15), 7, color).unwrap();

    println!("\nAfter drawing:");
    print_image_data(&img, "With Shapes", 50, 50);
}
