//! # OpenCV-Rust
//!
//! A pure Rust implementation of core OpenCV functionality.
//!
//! This library provides computer vision and image processing capabilities
//! inspired by OpenCV, implemented entirely in Rust for safety and performance.
//!
//! ## Features
//!
//! - **Core**: Basic data structures (Mat, Point, Size, Rect, Scalar)
//! - **Image I/O**: Reading and writing images in various formats
//! - **Image Processing**: Color conversion, filtering, geometric transformations
//! - **Thresholding**: Binary and adaptive thresholding
//!
//! ## Example
//!
//! ```rust,no_run
//! use opencv_rust::prelude::*;
//! use opencv_rust::imgcodecs::{imread, imwrite};
//! use opencv_rust::imgproc::cvt_color;
//!
//! # fn main() -> opencv_rust::error::Result<()> {
//! // Read an image
//! let src = imread("input.jpg")?;
//!
//! // Convert to grayscale
//! let mut gray = Mat::new(1, 1, 1, MatDepth::U8)?;
//! cvt_color(&src, &mut gray, ColorConversionCode::RgbToGray)?;
//!
//! // Save the result
//! imwrite("output.jpg", &gray)?;
//! # Ok(())
//! # }
//! ```

pub mod core;
pub mod error;
pub mod imgcodecs;
pub mod imgproc;
pub mod features2d;
pub mod video;
pub mod videoio;
pub mod ml;
pub mod objdetect;
pub mod photo;
pub mod calib3d;

pub mod prelude {
    //! Convenience module that re-exports commonly used items
    pub use crate::core::{Mat, MatDepth, Point, Point2f, Size, Rect, Scalar};
    pub use crate::core::types::{Point3f, ColorConversionCode, InterpolationFlag, ThresholdType};
    pub use crate::error::{Error, Result};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_basic_workflow() {
        // Create a mat
        let mat = Mat::new(100, 100, 3, MatDepth::U8).unwrap();
        assert_eq!(mat.rows(), 100);
        assert_eq!(mat.cols(), 100);

        // Create geometric types
        let p = Point::new(10, 20);
        let s = Size::new(640, 480);
        let r = Rect::new(0, 0, 100, 100);

        assert!(r.contains(p));
        assert_eq!(s.area(), 640 * 480);
    }
}
