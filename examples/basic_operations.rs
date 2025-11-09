use opencv_rust::prelude::*;
use opencv_rust::core::Mat;

fn main() -> Result<()> {
    println!("OpenCV-Rust Basic Operations Example");
    println!("====================================\n");

    // Create a matrix
    println!("Creating a 480x640 RGB image...");
    let mat = Mat::new(480, 640, 3, MatDepth::U8)?;
    println!("  Size: {}x{}", mat.cols(), mat.rows());
    println!("  Channels: {}", mat.channels());
    println!("  Total bytes: {}\n", mat.data().len());

    // Create geometric types
    println!("Working with geometric types:");
    let point = Point::new(100, 200);
    println!("  Point: ({}, {})", point.x, point.y);

    let size = Size::new(640, 480);
    println!("  Size: {}x{} (area: {})", size.width, size.height, size.area());

    let rect = Rect::new(50, 50, 200, 150);
    println!("  Rectangle: x={}, y={}, w={}, h={}", rect.x, rect.y, rect.width, rect.height);
    println!("  Contains point (100, 100)? {}", rect.contains(Point::new(100, 100)));

    // Scalar operations
    println!("\nScalar operations:");
    let s1 = Scalar::from_rgb(255, 128, 64);
    let s2 = Scalar::from_rgb(50, 50, 50);
    let sum = s1 + s2;
    println!("  RGB(255, 128, 64) + RGB(50, 50, 50) = RGB({}, {}, {})",
             sum.val[0] as u8, sum.val[1] as u8, sum.val[2] as u8);

    // Create colored image
    println!("\nCreating a blue image...");
    let blue_img = Mat::new_with_default(
        200,
        300,
        3,
        MatDepth::U8,
        Scalar::from_rgb(0, 0, 255)
    )?;
    println!("  Created {}x{} blue image", blue_img.cols(), blue_img.rows());

    // ROI extraction
    println!("\nExtracting region of interest...");
    let roi = blue_img.roi(Rect::new(50, 50, 100, 100))?;
    println!("  ROI size: {}x{}", roi.cols(), roi.rows());

    println!("\nBasic operations completed successfully!");

    Ok(())
}
