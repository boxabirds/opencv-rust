use opencv_rust::prelude::*;
use opencv_rust::imgproc::{
    cvt_color, resize, blur, gaussian_blur, threshold, sobel, canny,
    erode, dilate, get_structuring_element, MorphShape,
    find_contours, contour_area, bounding_rect, RetrievalMode, ChainApproxMode,
    calc_hist, equalize_hist, line, rectangle, circle
};
use opencv_rust::features2d::{harris_corners, good_features_to_track, fast, ORB};
use opencv_rust::ml::kmeans;

fn main() -> Result<()> {
    println!("=== OpenCV-Rust Comprehensive Demo ===\n");

    // 1. Create and manipulate matrices
    println!("1. Matrix Operations");
    let mut img = Mat::new(400, 600, 3, MatDepth::U8)?;

    // Fill with gradient
    let num_rows = img.rows();
    let num_cols = img.cols();
    for row in 0..num_rows {
        for col in 0..num_cols {
            let pixel = img.at_mut(row, col)?;
            pixel[0] = (col * 255 / num_cols) as u8;
            pixel[1] = (row * 255 / num_rows) as u8;
            pixel[2] = 128;
        }
    }
    println!("   Created {}x{} image with gradient\n", img.cols(), img.rows());

    // 2. Color space conversions
    println!("2. Color Space Conversions");
    let mut gray = Mat::new(1, 1, 1, MatDepth::U8)?;
    cvt_color(&img, &mut gray, ColorConversionCode::RgbToGray)?;
    println!("   Converted to grayscale: {}x{} (1 channel)\n", gray.cols(), gray.rows());

    // 3. Geometric transformations
    println!("3. Geometric Transformations");
    let mut resized = Mat::new(1, 1, 1, MatDepth::U8)?;
    resize(&gray, &mut resized, Size::new(300, 200), InterpolationFlag::Linear)?;
    println!("   Resized to {}x{}\n", resized.cols(), resized.rows());

    // 4. Image filtering
    println!("4. Image Filtering");
    let mut blurred = Mat::new(1, 1, 1, MatDepth::U8)?;
    blur(&gray, &mut blurred, Size::new(5, 5))?;
    println!("   Applied box blur (5x5)");

    let mut gaussian_blurred = Mat::new(1, 1, 1, MatDepth::U8)?;
    gaussian_blur(&gray, &mut gaussian_blurred, Size::new(5, 5), 1.5)?;
    println!("   Applied Gaussian blur (5x5, σ=1.5)\n");

    // 5. Edge detection
    println!("5. Edge Detection");
    let mut edges_sobel = Mat::new(1, 1, 1, MatDepth::U8)?;
    sobel(&gray, &mut edges_sobel, 1, 1, 3)?;
    println!("   Sobel edge detection");

    let mut edges_canny = Mat::new(1, 1, 1, MatDepth::U8)?;
    canny(&gray, &mut edges_canny, 50.0, 150.0)?;
    println!("   Canny edge detection\n");

    // 6. Morphological operations
    println!("6. Morphological Operations");
    let kernel = get_structuring_element(MorphShape::Rect, Size::new(5, 5));

    let mut eroded = Mat::new(1, 1, 1, MatDepth::U8)?;
    erode(&gray, &mut eroded, &kernel)?;
    println!("   Erosion applied");

    let mut dilated = Mat::new(1, 1, 1, MatDepth::U8)?;
    dilate(&gray, &mut dilated, &kernel)?;
    println!("   Dilation applied\n");

    // 7. Thresholding
    println!("7. Thresholding");
    let mut thresholded = Mat::new(1, 1, 1, MatDepth::U8)?;
    threshold(&gray, &mut thresholded, 128.0, 255.0, ThresholdType::Binary)?;
    println!("   Binary threshold (T=128)\n");

    // 8. Contour detection
    println!("8. Contour Detection");
    let contours = find_contours(&thresholded, RetrievalMode::External, ChainApproxMode::Simple)?;
    println!("   Found {} contours", contours.len());

    if let Some(first_contour) = contours.first() {
        let area = contour_area(first_contour);
        let bbox = bounding_rect(first_contour);
        println!("   First contour: area={:.2}, bbox={}x{}\n", area, bbox.width, bbox.height);
    }

    // 9. Histogram operations
    println!("9. Histogram Operations");
    let hist = calc_hist(&gray, 256, (0.0, 256.0))?;
    let max_bin = hist.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
    println!("   Histogram calculated: max bin {} with value {:.0}", max_bin.0, max_bin.1);

    let mut equalized = Mat::new(1, 1, 1, MatDepth::U8)?;
    equalize_hist(&gray, &mut equalized)?;
    println!("   Histogram equalization applied\n");

    // 10. Drawing functions
    println!("10. Drawing Functions");
    let mut canvas = Mat::new(400, 600, 3, MatDepth::U8)?;

    line(&mut canvas, Point::new(50, 50), Point::new(550, 350), Scalar::from_rgb(255, 0, 0), 2)?;
    println!("   Drew line");

    rectangle(&mut canvas, Rect::new(100, 100, 200, 150), Scalar::from_rgb(0, 255, 0), 3)?;
    println!("   Drew rectangle");

    circle(&mut canvas, Point::new(300, 200), 50, Scalar::from_rgb(0, 0, 255))?;
    println!("   Drew circle\n");

    // 11. Feature detection
    println!("11. Feature Detection");

    // Harris corners
    let corners = harris_corners(&gray, 3, 3, 0.04, 10000.0)?;
    println!("   Harris corner detection: {} corners", corners.len());

    // Good features to track
    let good_features = good_features_to_track(&gray, 100, 0.01, 10.0, 3)?;
    println!("   Good features to track: {} features", good_features.len());

    // FAST keypoints
    let fast_keypoints = fast(&gray, 20, true)?;
    println!("   FAST keypoint detection: {} keypoints\n", fast_keypoints.len());

    // 12. Feature descriptors (ORB)
    println!("12. Feature Descriptors");
    let orb = ORB::new(50);
    let (keypoints, descriptors) = orb.detect_and_compute(&gray)?;
    println!("   ORB: {} keypoints with {} descriptors\n", keypoints.len(), descriptors.len());

    // 13. Machine Learning - K-means clustering
    println!("13. Machine Learning");
    let data = vec![
        vec![1.0, 1.0],
        vec![1.5, 2.0],
        vec![3.0, 4.0],
        vec![5.0, 7.0],
        vec![3.5, 5.0],
        vec![4.5, 5.0],
        vec![3.5, 4.5],
    ];

    let mut labels = vec![0; data.len()];
    let (centers, compactness) = kmeans(
        &data,
        2,
        &mut labels,
        100,
        1.0,
        opencv_rust::ml::KMeansFlags::PPCenters,
    )?;

    println!("   K-means clustering:");
    println!("     {} clusters, compactness={:.2}", centers.len(), compactness);
    println!("     Labels: {:?}\n", labels);

    // 14. Core operations
    println!("14. Core Matrix Operations");
    use opencv_rust::core::{add, split, merge, mean};

    let mat1 = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(100.0))?;
    let mat2 = Mat::new_with_default(100, 100, 3, MatDepth::U8, Scalar::all(50.0))?;

    let mut result = Mat::new(1, 1, 1, MatDepth::U8)?;
    add(&mat1, &mat2, &mut result)?;
    println!("   Matrix addition: 100 + 50 = {}", result.at(50, 50)?[0]);

    let channels = split(&img)?;
    println!("   Split {} channels", channels.len());

    let mut merged = Mat::new(1, 1, 1, MatDepth::U8)?;
    merge(&channels, &mut merged)?;
    println!("   Merged back to {} channels", merged.channels());

    let avg = mean(&gray)?;
    println!("   Mean pixel value: {:.2}\n", avg.val[0]);

    println!("=== Demo Complete! ===");
    println!("\nImplemented {} major OpenCV modules:", 9);
    println!("  • core      - Matrix operations, geometric types");
    println!("  • imgproc   - Image processing, filtering, morphology");
    println!("  • imgcodecs - Image I/O");
    println!("  • features2d - Feature detection and description");
    println!("  • video     - Optical flow, tracking");
    println!("  • ml        - Machine learning (k-means, SVM)");
    println!("  • objdetect - Object detection (HOG, cascades)");
    println!("  • photo     - Computational photography");
    println!("\nAll operations completed successfully!");

    Ok(())
}
