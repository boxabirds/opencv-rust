//! Test utilities for WASM-based OpenCV.js parity tests
//!
//! This module provides helper functions and utilities for testing
//! WASM bindings against OpenCV.js behavior.

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use wasm_bindgen_test::*;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use opencv_rust::wasm::WasmMat;

/// Create a simple test image (grayscale, 10x10 pixels)
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub fn create_test_image_gray() -> WasmMat {
    let width = 10;
    let height = 10;
    let channels = 1;

    let mut data = vec![0u8; width * height * channels];

    // Create a simple gradient pattern
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            data[idx] = ((x + y) * 25) as u8;
        }
    }

    WasmMat::from_image_data(&data, width, height, channels)
        .expect("Failed to create test image")
}

/// Create a test image with RGB channels (10x10 pixels)
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub fn create_test_image_rgb() -> WasmMat {
    let width = 10;
    let height = 10;
    let channels = 3;

    let mut data = vec![0u8; width * height * channels];

    // Create a simple RGB gradient
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * channels;
            data[idx] = (x * 25) as u8;     // R
            data[idx + 1] = (y * 25) as u8; // G
            data[idx + 2] = 128;            // B (constant)
        }
    }

    WasmMat::from_image_data(&data, width, height, channels)
        .expect("Failed to create RGB test image")
}

/// Create a larger test image for performance testing (100x100 pixels)
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub fn create_test_image_large() -> WasmMat {
    let width = 100;
    let height = 100;
    let channels = 3;

    let mut data = vec![0u8; width * height * channels];

    // Create a checkerboard pattern
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * channels;
            let value = if (x / 10 + y / 10) % 2 == 0 { 255 } else { 0 };
            data[idx] = value;
            data[idx + 1] = value;
            data[idx + 2] = value;
        }
    }

    WasmMat::from_image_data(&data, width, height, channels)
        .expect("Failed to create large test image")
}

/// Compare two images and check if they're similar within a tolerance
///
/// Returns true if the average pixel difference is within the tolerance
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub fn images_are_similar(img1: &WasmMat, img2: &WasmMat, tolerance: f64) -> bool {
    // Check dimensions match
    if img1.width() != img2.width() || img1.height() != img2.height() || img1.channels() != img2.channels() {
        return false;
    }

    let data1 = img1.get_data();
    let data2 = img2.get_data();

    if data1.len() != data2.len() {
        return false;
    }

    // Calculate average absolute difference
    let total_diff: i32 = data1.iter()
        .zip(data2.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .sum();

    let avg_diff = total_diff as f64 / data1.len() as f64;

    avg_diff <= tolerance
}

/// Check if an image has the expected dimensions
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub fn check_dimensions(img: &WasmMat, expected_width: usize, expected_height: usize, expected_channels: usize) -> bool {
    img.width() == expected_width &&
    img.height() == expected_height &&
    img.channels() == expected_channels
}

/// Calculate the average pixel value of an image
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub fn average_pixel_value(img: &WasmMat) -> f64 {
    let data = img.get_data();
    let sum: u64 = data.iter().map(|&x| x as u64).sum();
    sum as f64 / data.len() as f64
}

/// Calculate the standard deviation of pixel values
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub fn pixel_stddev(img: &WasmMat) -> f64 {
    let data = img.get_data();
    let mean = average_pixel_value(img);

    let variance: f64 = data.iter()
        .map(|&x| {
            let diff = x as f64 - mean;
            diff * diff
        })
        .sum::<f64>() / data.len() as f64;

    variance.sqrt()
}

/// Check if an image is completely black (all zeros)
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub fn is_black(img: &WasmMat) -> bool {
    img.get_data().iter().all(|&x| x == 0)
}

/// Check if an image is completely white (all 255)
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub fn is_white(img: &WasmMat) -> bool {
    img.get_data().iter().all(|&x| x == 255)
}

/// Count non-zero pixels in an image
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
pub fn count_nonzero(img: &WasmMat) -> usize {
    img.get_data().iter().filter(|&&x| x != 0).count()
}

#[cfg(test)]
mod tests {
    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    use super::*;

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    #[wasm_bindgen_test]
    fn test_create_test_image_gray() {
        let img = create_test_image_gray();
        assert_eq!(img.width(), 10);
        assert_eq!(img.height(), 10);
        assert_eq!(img.channels(), 1);
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    #[wasm_bindgen_test]
    fn test_create_test_image_rgb() {
        let img = create_test_image_rgb();
        assert_eq!(img.width(), 10);
        assert_eq!(img.height(), 10);
        assert_eq!(img.channels(), 3);
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    #[wasm_bindgen_test]
    fn test_images_are_similar() {
        let img1 = create_test_image_gray();
        let img2 = create_test_image_gray();
        assert!(images_are_similar(&img1, &img2, 0.0));
    }

    #[cfg(all(target_arch = "wasm32", feature = "wasm"))]
    #[wasm_bindgen_test]
    fn test_check_dimensions() {
        let img = create_test_image_gray();
        assert!(check_dimensions(&img, 10, 10, 1));
        assert!(!check_dimensions(&img, 20, 20, 1));
    }
}
