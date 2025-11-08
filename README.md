# OpenCV-Rust

A comprehensive pure Rust implementation of OpenCV (Open Source Computer Vision Library) functionality.

Over **6,000+ lines** of pure Rust code implementing the major OpenCV modules without any C/C++ dependencies.

## Features

This library provides a complete Rust-native implementation of essential OpenCV operations across 9 major modules:

### 🎯 Core Module
- Matrix operations, geometric types, channel operations, statistical functions

### 🖼️ Image I/O
- Read/write images in PNG, JPEG, BMP formats

### 🎨 Image Processing
- Color conversions, geometric transformations, filtering, edge detection
- Morphological operations, thresholding, contour detection
- Histogram operations, Hough transforms, drawing functions

### 🔍 Features2D
- Keypoint detection (Harris, Shi-Tomasi, FAST)
- Feature descriptors (ORB, BRIEF, SIFT)
- Feature matching (Brute Force, k-NN, radius matching)

### 📹 Video Analysis
- Optical flow (Lucas-Kanade, Farneback)
- Object tracking (Mean Shift, CAMShift, background subtraction)

### 🤖 Machine Learning
- K-means clustering
- SVM (Linear, RBF, Poly, Sigmoid kernels)

### 🎯 Object Detection
- HOG descriptor
- Cascade classifiers (Haar-like features)

### 📷 Computational Photography
- Inpainting, non-local means denoising

## Quick Start

```rust
use opencv_rust::prelude::*;
use opencv_rust::imgproc::{cvt_color, canny};
use opencv_rust::features2d::ORB;

fn main() -> Result<()> {
    let img = Mat::new(480, 640, 3, MatDepth::U8)?;

    let mut gray = Mat::new(1, 1, 1, MatDepth::U8)?;
    cvt_color(&img, &mut gray, ColorConversionCode::RgbToGray)?;

    let mut edges = Mat::new(1, 1, 1, MatDepth::U8)?;
    canny(&gray, &mut edges, 50.0, 150.0)?;

    let orb = ORB::new(500);
    let (keypoints, descriptors) = orb.detect_and_compute(&gray)?;

    Ok(())
}
```

## Examples

```bash
cargo run --example comprehensive_demo
```

## Statistics

- **Lines of Code**: 6,000+
- **Modules**: 9
- **Functions**: 100+
- **Tests**: 53 (all passing)
- **Zero unsafe code**

## License

Apache License 2.0

