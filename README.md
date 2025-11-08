# OpenCV-Rust

A pure Rust implementation of core OpenCV functionality for computer vision and image processing.

## Features

This library provides a Rust-native implementation of essential OpenCV operations:

### Core Module
- **Mat**: The fundamental matrix/image data structure
- **Geometric Types**: Point, Point2f, Size, Rect
- **Scalar**: Multi-channel scalar values
- Support for multiple data types (U8, U16, F32, F64)

### Image I/O (imgcodecs)
- `imread()`: Load images from disk (supports PNG, JPEG, BMP, and more)
- `imwrite()`: Save images to disk

### Image Processing (imgproc)

#### Color Conversions
- RGB ↔ BGR
- RGB/BGR → Grayscale
- Grayscale → RGB/BGR
- RGB/BGR ↔ HSV

#### Geometric Transformations
- `resize()`: Resize images with multiple interpolation methods
  - Nearest neighbor
  - Bilinear interpolation
- `flip()`: Flip images horizontally, vertically, or both

#### Filtering
- `blur()`: Box filter (simple averaging)
- `gaussian_blur()`: Gaussian blur with configurable kernel and sigma
- `median_blur()`: Median filtering for noise reduction

#### Thresholding
- `threshold()`: Binary, binary inverted, truncate, to-zero, to-zero inverted
- `adaptive_threshold()`: Adaptive thresholding with mean or Gaussian methods

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
opencv-rust = "0.1.0"
```

## Quick Start

```rust
use opencv_rust::prelude::*;
use opencv_rust::imgcodecs::{imread, imwrite};
use opencv_rust::imgproc::{cvt_color, resize, gaussian_blur};
use opencv_rust::imgproc::{ColorConversionCode, InterpolationFlag};

fn main() -> Result<()> {
    // Load an image
    let img = imread("input.jpg")?;

    // Convert to grayscale
    let mut gray = Mat::new(1, 1, 1, MatDepth::U8)?;
    cvt_color(&img, &mut gray, ColorConversionCode::RgbToGray)?;

    // Resize the image
    let mut resized = Mat::new(1, 1, 1, MatDepth::U8)?;
    resize(&gray, &mut resized, Size::new(800, 600), InterpolationFlag::Linear)?;

    // Apply Gaussian blur
    let mut blurred = Mat::new(1, 1, 1, MatDepth::U8)?;
    gaussian_blur(&resized, &mut blurred, Size::new(5, 5), 1.5)?;

    // Save the result
    imwrite("output.jpg", &blurred)?;

    Ok(())
}
```

## Examples

The `examples/` directory contains several demonstration programs:

### Basic Operations
```bash
cargo run --example basic_operations
```

Demonstrates:
- Creating matrices
- Working with geometric types (Point, Size, Rect)
- Scalar operations
- Region of Interest (ROI) extraction

### Image Processing
```bash
cargo run --example image_processing
```

Demonstrates:
- Color space conversions
- Image resizing
- Various blur filters
- Thresholding operations

## Architecture

OpenCV-Rust is designed with the following principles:

1. **Pure Rust**: No C/C++ dependencies, eliminating build complexity
2. **Type Safety**: Leverages Rust's type system for safer image processing
3. **Memory Safety**: No manual memory management, leveraging Rust's ownership system
4. **Performance**: Uses efficient algorithms and data structures (ndarray, separable filters)
5. **Ergonomic API**: Familiar interface for OpenCV users

## Performance Considerations

- Separable filters (Gaussian blur) use two 1D convolutions instead of one 2D for efficiency
- In-place operations where possible to reduce memory allocations
- Uses `ndarray` for efficient multi-dimensional array operations

## Limitations and Future Work

Current limitations:
- Only U8 depth fully supported for image processing operations
- Limited interpolation methods (nearest neighbor and bilinear)
- No GPU acceleration yet

Planned features:
- More advanced feature detection (Harris corners, SIFT, etc.)
- Morphological operations (erode, dilate, etc.)
- Contour detection and analysis
- Drawing functions (line, circle, rectangle, text)
- Video I/O support
- More sophisticated interpolation (cubic, Lanczos)
- SIMD optimizations

## Contributing

Contributions are welcome! Areas of interest:
- Additional image processing operations
- Performance optimizations
- Bug fixes and tests
- Documentation improvements

## License

Licensed under the Apache License 2.0. See LICENSE file for details.

## Comparison with opencv-rust bindings

This project differs from the `opencv-rust` bindings crate:

| Feature | opencv-rust (this) | opencv-rust (bindings) |
|---------|-------------------|------------------------|
| Implementation | Pure Rust | C++ bindings |
| Build complexity | Cargo only | Requires OpenCV C++ installation |
| Safety | Full Rust safety | Unsafe FFI |
| Coverage | Core features | Comprehensive OpenCV API |
| Performance | Good (pure Rust) | Excellent (optimized C++) |

Use this library if you want:
- Simpler builds without system dependencies
- Pure Rust environment
- Core image processing functionality
- Educational purposes

Use opencv-rust bindings if you need:
- Full OpenCV feature set
- Maximum performance
- Existing OpenCV ecosystem

## Acknowledgments

Inspired by the excellent OpenCV library. Algorithm implementations based on OpenCV documentation and computer vision literature.
