use opencv_rust::prelude::*;
use opencv_rust::imgproc::{cvt_color, resize, blur, gaussian_blur, threshold};

fn main() -> Result<()> {
    println!("OpenCV-Rust Image Processing Example");
    println!("====================================\n");

    // For demonstration, create a synthetic image
    println!("Creating a test image...");
    let mut src = Mat::new(400, 600, 3, MatDepth::U8)?;

    // Fill with a gradient pattern
    let num_rows = src.rows();
    let num_cols = src.cols();
    for row in 0..num_rows {
        for col in 0..num_cols {
            let pixel = src.at_mut(row, col)?;
            pixel[0] = (col * 255 / num_cols) as u8; // Red channel
            pixel[1] = (row * 255 / num_rows) as u8; // Green channel
            pixel[2] = 128; // Blue channel
        }
    }

    println!("Original image: {}x{} with {} channels\n",
             src.cols(), src.rows(), src.channels());

    // Color conversion
    println!("Converting to grayscale...");
    let mut gray = Mat::new(1, 1, 1, MatDepth::U8)?;
    cvt_color(&src, &mut gray, ColorConversionCode::RgbToGray)?;
    println!("  Grayscale image: {}x{} with {} channel\n",
             gray.cols(), gray.rows(), gray.channels());

    // Resize
    println!("Resizing image to 300x200...");
    let mut resized = Mat::new(1, 1, 1, MatDepth::U8)?;
    resize(&src, &mut resized, Size::new(300, 200), InterpolationFlag::Linear)?;
    println!("  Resized to: {}x{}\n", resized.cols(), resized.rows());

    // Blur
    println!("Applying box blur (5x5 kernel)...");
    let mut blurred = Mat::new(1, 1, 1, MatDepth::U8)?;
    blur(&src, &mut blurred, Size::new(5, 5))?;
    println!("  Blur applied\n");

    // Gaussian blur
    println!("Applying Gaussian blur (5x5 kernel, sigma=1.5)...");
    let mut gaussian_blurred = Mat::new(1, 1, 1, MatDepth::U8)?;
    gaussian_blur(&src, &mut gaussian_blurred, Size::new(5, 5), 1.5)?;
    println!("  Gaussian blur applied\n");

    // Threshold
    println!("Applying binary threshold (thresh=128, maxval=255)...");
    let mut thresholded = Mat::new(1, 1, 1, MatDepth::U8)?;
    threshold(&gray, &mut thresholded, 128.0, 255.0, ThresholdType::Binary)?;
    println!("  Threshold applied\n");

    // Channel swap
    println!("Converting RGB to BGR...");
    let mut bgr = Mat::new(1, 1, 1, MatDepth::U8)?;
    cvt_color(&src, &mut bgr, ColorConversionCode::RgbToBgr)?;
    println!("  Channels swapped\n");

    println!("All image processing operations completed successfully!");
    println!("\nNote: To see actual results, use imread() to load an image");
    println!("and imwrite() to save the processed results.");

    Ok(())
}
