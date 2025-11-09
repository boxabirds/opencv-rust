/// Test utilities for bit-level accuracy validation
use opencv_rust::core::{Mat, MatDepth};
use std::fmt::Write as FmtWrite;

/// Compare two images pixel-by-pixel with detailed error reporting
pub fn assert_images_equal(actual: &Mat, expected: &Mat, test_name: &str) {
    assert_eq!(actual.rows(), expected.rows(),
        "{}: Image dimensions mismatch (rows)", test_name);
    assert_eq!(actual.cols(), expected.cols(),
        "{}: Image dimensions mismatch (cols)", test_name);
    assert_eq!(actual.channels(), expected.channels(),
        "{}: Channel count mismatch", test_name);
    assert_eq!(actual.depth(), expected.depth(),
        "{}: Depth mismatch", test_name);

    let mut diff_count = 0;
    let mut max_diff = 0u32;
    let mut first_diff_loc = None;
    let mut diff_details = String::new();

    for row in 0..actual.rows() {
        for col in 0..actual.cols() {
            let actual_pixel = actual.at(row, col).unwrap();
            let expected_pixel = expected.at(row, col).unwrap();

            for ch in 0..actual.channels() {
                let actual_val = actual_pixel[ch];
                let expected_val = expected_pixel[ch];

                if actual_val != expected_val {
                    let diff = (actual_val as i32 - expected_val as i32).abs() as u32;
                    diff_count += 1;
                    max_diff = max_diff.max(diff);

                    if first_diff_loc.is_none() {
                        first_diff_loc = Some((row, col, ch));
                        writeln!(
                            &mut diff_details,
                            "First diff at ({}, {}) ch{}: actual={}, expected={}, diff={}",
                            row, col, ch, actual_val, expected_val, diff
                        ).unwrap();
                    }

                    // Report first 10 diffs
                    if diff_count <= 10 {
                        writeln!(
                            &mut diff_details,
                            "  ({}, {}) ch{}: {} vs {} (diff: {})",
                            row, col, ch, actual_val, expected_val, diff
                        ).unwrap();
                    }
                }
            }
        }
    }

    if diff_count > 0 {
        let total_pixels = actual.rows() * actual.cols() * actual.channels();
        panic!(
            "\n{}: Pixel differences found!\n\
             Total pixels: {}\n\
             Differing pixels: {} ({:.2}%)\n\
             Max difference: {}\n\
             {}\n\
             {}",
            test_name,
            total_pixels,
            diff_count,
            (diff_count as f64 / total_pixels as f64) * 100.0,
            max_diff,
            if diff_count > 10 { format!("... and {} more differences\n", diff_count - 10) } else { String::new() },
            diff_details
        );
    }
}

/// Compare two images with tolerance for floating-point rounding differences
pub fn assert_images_near(actual: &Mat, expected: &Mat, max_diff: u8, test_name: &str) {
    assert_eq!(actual.rows(), expected.rows(),
        "{}: Image dimensions mismatch (rows)", test_name);
    assert_eq!(actual.cols(), expected.cols(),
        "{}: Image dimensions mismatch (cols)", test_name);
    assert_eq!(actual.channels(), expected.channels(),
        "{}: Channel count mismatch", test_name);

    let mut excessive_diff_count = 0;
    let mut first_excessive_diff = None;

    for row in 0..actual.rows() {
        for col in 0..actual.cols() {
            let actual_pixel = actual.at(row, col).unwrap();
            let expected_pixel = expected.at(row, col).unwrap();

            for ch in 0..actual.channels() {
                let actual_val = actual_pixel[ch];
                let expected_val = expected_pixel[ch];
                let diff = (actual_val as i32 - expected_val as i32).abs() as u8;

                if diff > max_diff {
                    excessive_diff_count += 1;
                    if first_excessive_diff.is_none() {
                        first_excessive_diff = Some((row, col, ch, actual_val, expected_val, diff));
                    }
                }
            }
        }
    }

    if excessive_diff_count > 0 {
        let (row, col, ch, actual_val, expected_val, diff) = first_excessive_diff.unwrap();
        let total_pixels = actual.rows() * actual.cols() * actual.channels();
        panic!(
            "\n{}: Excessive pixel differences found!\n\
             Tolerance: {}\n\
             Total pixels: {}\n\
             Pixels exceeding tolerance: {} ({:.2}%)\n\
             First excessive diff at ({}, {}) ch{}: actual={}, expected={}, diff={}",
            test_name,
            max_diff,
            total_pixels,
            excessive_diff_count,
            (excessive_diff_count as f64 / total_pixels as f64) * 100.0,
            row, col, ch, actual_val, expected_val, diff
        );
    }
}

/// Compute statistics about pixel differences
pub fn compute_diff_stats(actual: &Mat, expected: &Mat) -> DiffStats {
    let mut total_diff = 0u64;
    let mut max_diff = 0u32;
    let mut diff_count = 0;
    let total_pixels = (actual.rows() * actual.cols() * actual.channels()) as u64;

    for row in 0..actual.rows() {
        for col in 0..actual.cols() {
            let actual_pixel = actual.at(row, col).unwrap();
            let expected_pixel = expected.at(row, col).unwrap();

            for ch in 0..actual.channels() {
                let diff = (actual_pixel[ch] as i32 - expected_pixel[ch] as i32).abs() as u32;
                if diff > 0 {
                    diff_count += 1;
                    total_diff += diff as u64;
                    max_diff = max_diff.max(diff);
                }
            }
        }
    }

    DiffStats {
        total_pixels,
        diff_count,
        max_diff,
        mean_diff: if diff_count > 0 { total_diff as f64 / diff_count as f64 } else { 0.0 },
        percent_different: (diff_count as f64 / total_pixels as f64) * 100.0,
    }
}

#[derive(Debug)]
pub struct DiffStats {
    pub total_pixels: u64,
    pub diff_count: u64,
    pub max_diff: u32,
    pub mean_diff: f64,
    pub percent_different: f64,
}

impl std::fmt::Display for DiffStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DiffStats {{ total: {}, different: {} ({:.3}%), max: {}, mean: {:.2} }}",
            self.total_pixels,
            self.diff_count,
            self.percent_different,
            self.max_diff,
            self.mean_diff
        )
    }
}

/// Print detailed image data for debugging
pub fn print_image_data(img: &Mat, name: &str, max_rows: usize, max_cols: usize) {
    println!("\n{} ({}x{}, {} channels):", name, img.rows(), img.cols(), img.channels());

    let rows = img.rows().min(max_rows);
    let cols = img.cols().min(max_cols);

    for row in 0..rows {
        print!("  Row {}: ", row);
        for col in 0..cols {
            let pixel = img.at(row, col).unwrap();
            if img.channels() == 1 {
                print!("{:3} ", pixel[0]);
            } else {
                print!("[");
                for ch in 0..img.channels() {
                    print!("{:3}", pixel[ch]);
                    if ch < img.channels() - 1 {
                        print!(",");
                    }
                }
                print!("] ");
            }
        }
        println!();
    }

    if img.rows() > max_rows || img.cols() > max_cols {
        println!("  ... (truncated)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv_rust::core::types::Scalar;

    #[test]
    fn test_assert_images_equal_identical() {
        let img1 = Mat::new_with_default(10, 10, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let img2 = Mat::new_with_default(10, 10, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        assert_images_equal(&img1, &img2, "identical images");
    }

    #[test]
    #[should_panic(expected = "Pixel differences found")]
    fn test_assert_images_equal_different() {
        let img1 = Mat::new_with_default(10, 10, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
        let mut img2 = Mat::new_with_default(10, 10, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

        img2.at_mut(5, 5).unwrap()[0] = 129;

        assert_images_equal(&img1, &img2, "different images");
    }

    #[test]
    fn test_diff_stats() {
        let img1 = Mat::new_with_default(10, 10, 1, MatDepth::U8, Scalar::all(100.0)).unwrap();
        let img2 = Mat::new_with_default(10, 10, 1, MatDepth::U8, Scalar::all(105.0)).unwrap();

        let stats = compute_diff_stats(&img1, &img2);

        assert_eq!(stats.diff_count, 100); // All pixels differ
        assert_eq!(stats.max_diff, 5);
        assert!((stats.mean_diff - 5.0).abs() < 0.01);
    }
}
