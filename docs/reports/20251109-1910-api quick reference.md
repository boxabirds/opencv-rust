# OpenCV-Rust Public API - Quick Reference

## Core Types and Patterns at a Glance

### Essential Data Types
```rust
// Matrix (main data structure)
let mat = Mat::new(height, width, channels, MatDepth::U8)?;
let pixel = mat.at(row, col)?;  // Get pixel
mat.at_mut(row, col)?[0] = 255; // Set pixel

// Geometric Types
let point = Point::new(x, y);
let size = Size::new(width, height);
let rect = Rect::new(x, y, width, height);
let scalar = Scalar::from_rgb(255, 128, 64);
```

### Naming Conventions at a Glance

| Item | Pattern | Examples |
|------|---------|----------|
| **Modules** | snake_case | imgproc, features2d, ml |
| **Structs** | PascalCase | Mat, ORB, SVM, RandomForest |
| **Functions** | snake_case | gaussian_blur, find_contours, threshold |
| **Methods** | snake_case | .rows(), .detect_and_compute(), .predict() |
| **Enums** | PascalCase | MatDepth, ColorConversionCode, ThresholdType |
| **Enum Variants** | PascalCase | MatDepth::U8, ThresholdType::Binary |
| **Builder Methods** | .with_*() | .with_threshold(), .with_scale_factor() |
| **Getters** | direct | .rows(), .channels(), .data() |
| **Setters** | .set_*() | .set_to(), .set_f32() |
| **Queries** | .is_*() | .is_opened(), .is_empty() |

---

## Most Common Public APIs

### Image Processing (imgproc)
```rust
use opencv_rust::imgproc::*;

// Color
cvt_color(&src, &mut dst, ColorConversionCode::RgbToGray)?;

// Filtering
gaussian_blur(&src, &mut dst, Size::new(5, 5), 1.5)?;
blur(&src, &mut dst, Size::new(3, 3))?;
median_blur(&src, &mut dst, 5)?;

// Geometric
resize(&src, &mut dst, Size::new(300, 200), InterpolationFlag::Linear)?;
rotate(&src, &mut dst, RotateCode::Rotate90Clockwise)?;
flip(&src, &mut dst, 1)?;

// Edge Detection
canny(&src, &mut dst, 50.0, 150.0)?;
sobel(&src, &mut dst, 1, 0)?;
laplacian(&src, &mut dst, 3)?;

// Thresholding
threshold(&src, &mut dst, 127.0, 255.0, ThresholdType::Binary)?;

// Morphology
let kernel = get_structuring_element(MorphShape::Ellipse, Size::new(5, 5));
erode(&src, &mut dst, &kernel)?;
dilate(&src, &mut dst, &kernel)?;

// Contours
let contours = find_contours(&binary_image, RetrievalMode::Tree, ChainApproxMode::ApproxSimple)?;
for contour in contours {
    let area = contour_area(&contour);
    let perimeter = arc_length(&contour, true);
}

// Drawing
line(&mut img, pt1, pt2, Scalar::from_rgb(0, 255, 0), 2)?;
rectangle(&mut img, rect, Scalar::from_rgb(255, 0, 0), 2)?;
circle(&mut img, center, radius, Scalar::from_rgb(0, 0, 255))?;
```

### Feature Detection (features2d)
```rust
use opencv_rust::features2d::*;

// ORB
let orb = ORB::new(500);
let (keypoints, descriptors) = orb.detect_and_compute(&image)?;

// AKAZE
let akaze = AKAZE::new().with_threshold(0.001);
let (keypoints, descriptors) = akaze.detect_and_compute(&image)?;

// Matching
let matcher = BFMatcher::new(DistanceType::Hamming, false);
let matches = matcher.match_descriptors(&desc1, &desc2)?;
let good_matches = ratio_test_filter(&knn_matches, 0.7);
```

### Machine Learning (ml)
```rust
use opencv_rust::ml::*;

// SVM
let mut svm = SVM::new(SVMType::CSvc, SVMKernelType::Rbf);
svm.train(&training_data, &labels)?;
let prediction = svm.predict(&sample)?;

// Random Forest
let rf = RandomForest::classifier(100)
    .with_max_depth(20)
    .with_min_samples_split(5);
rf.train(&training_data, &labels)?;
let prediction = rf.predict(&sample)?;
let importances = rf.feature_importances(10);

// Decision Tree
let dt = DecisionTree::classifier()
    .with_max_depth(15);
dt.train(&training_data, &labels)?;

// KNearest
let knn = KNearest::classifier(5)
    .with_algorithm(Algorithm::KdTree);
knn.train(&training_data, &labels)?;
let neighbors = knn.find_nearest(&sample, 5)?;

// KMeans
let (centers, labels) = kmeans(&data, 3, 100, 0.01)?;
```

### Video I/O (videoio)
```rust
use opencv_rust::videoio::*;

// Read Video
let mut cap = VideoCapture::from_file("video.mp4")?;
let mut frame = Mat::new(1, 1, 1, MatDepth::U8)?;
while cap.read(&mut frame)? {
    // Process frame
}

// Write Video
let mut writer = VideoWriter::new(
    "output.mp4",
    FourCC::MP4V,
    30.0,
    (640, 480)
)?;
writer.write(&frame)?;
```

### Image I/O (imgcodecs)
```rust
use opencv_rust::imgcodecs::*;

let img = imread("input.jpg")?;
imwrite("output.jpg", &img)?;
```

### Core Operations (core)
```rust
use opencv_rust::core::*;

// Matrix operations
add(&a, &b, &mut dst)?;
subtract(&a, &b, &mut dst)?;
multiply(&a, &b, &mut dst, 2.0)?;

// Bitwise
bitwise_and(&a, &b, &mut dst)?;
bitwise_or(&a, &b, &mut dst)?;
bitwise_not(&a, &mut dst)?;

// Channel operations
let channels = split(&src)?;  // Split into channel matrices
merge(&channels, &mut dst)?;  // Merge channels back

// Statistics
let mean_val = mean(&src)?;
let (min_val, max_val, min_loc, max_loc) = min_max_loc(&src)?;
```

---

## Pattern Reference

### Constructor Pattern
```rust
// Basic constructor
let mat = Mat::new(rows, cols, channels, MatDepth::U8)?;

// Constructor with default value
let mat = Mat::new_with_default(rows, cols, channels, MatDepth::U8, Scalar::all(128.0))?;

// From raw data
let mat = Mat::from_raw(data, rows, cols, channels, MatDepth::U8)?;
```

### Builder Pattern
```rust
let detector = RandomForest::classifier(100)
    .with_max_depth(20)
    .with_min_samples_split(5)
    .with_max_features(MaxFeatures::Sqrt);
```

### In-Place Operations
```rust
// Functions modify destination matrix in place
let mut dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;
gaussian_blur(&src, &mut dst, Size::new(5, 5), 1.5)?;
```

### Error Handling
```rust
// All operations return Result<T>
match gaussian_blur(&src, &mut dst, Size::new(5, 5), 1.5) {
    Ok(()) => println!("Success"),
    Err(e) => eprintln!("Error: {:?}", e),
}

// Using ? operator
let result = threshold(&src, &mut dst, 128.0, 255.0, ThresholdType::Binary)?;
```

---

## Module Structure

```
opencv_rust/
├── core/           // Mat, Point, Size, Rect, basic operations
├── imgproc/        // Image processing (filter, threshold, edge, etc.)
├── imgcodecs/      // imread, imwrite
├── features2d/     // ORB, AKAZE, BRIEF, matching
├── video/          // Video tracking, background subtraction
├── videoio/        // VideoCapture, VideoWriter
├── ml/             // SVM, RandomForest, DecisionTree, KNearest, ANN, KMeans
├── objdetect/      // CascadeClassifier, QR detector, HOG
├── photo/          // Denoising, super-resolution, denoising
├── calib3d/        // Camera calibration, stereo vision
├── dnn/            // Neural networks, layer definitions
├── flann/          // KDTree, LSHIndex, nearest neighbor search
├── stitching/      // Image stitching, panorama
├── shape/          // Shape descriptors, contour analysis
├── gpu/            // GPU-accelerated operations (optional)
├── wasm/           // WebAssembly support (optional)
└── prelude/        // Common re-exports
```

---

## Performance Tips

1. **Use GPU acceleration when available**
   ```rust
   #[cfg(feature = "gpu")]
   {
       if gpu_available() {
           crate::gpu::ops::gaussian_blur_gpu(&src, &mut dst, ...)?;
       }
   }
   ```

2. **Use in-place operations for memory efficiency**
   ```rust
   // Better than creating new mats
   blur(&src, &mut dst, kernel)?;
   ```

3. **Leverage builder patterns for optimal parameters**
   ```rust
   let rf = RandomForest::classifier(100)
       .with_max_depth(optimal_depth)
       .with_max_features(optimal_features);
   ```

4. **Parallel processing with rayon feature**
   - Automatically enabled for CPU-heavy operations
   - median_blur, separable filters use parallelization

5. **Use WASM async operations for web**
   ```rust
   #[cfg(target_arch = "wasm32")]
   {
       let result = gaussian_blur_wasm(&src, Size::new(5, 5), 1.5).await?;
   }
   ```

---

## Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `UnsupportedOperation` | Wrong data type | Check `MatDepth`, ensure U8 for most operations |
| `InvalidParameter` | Bad kernel size | Kernel size must be odd |
| `InvalidParameter` | Mismatched matrix sizes | Ensure input/output sizes are compatible |
| `UnsupportedOperation` | GPU not available | Fall back to CPU or check `gpu_available()` |

---

## API Stability Notes

1. **Core Types** (stable): Mat, Point, Size, Rect, Scalar
2. **Image Processing** (stable): blur, gaussian_blur, canny, threshold
3. **Feature Detection** (stable): ORB, AKAZE, BRIEF matching
4. **ML Algorithms** (stable): SVM, RandomForest, KNearest
5. **Video I/O** (stable): VideoCapture, VideoWriter
6. **GPU Operations** (optional): May have breaking changes
7. **WASM Support** (experimental): API may change

---

## Useful Links in Code

- View detailed analysis: `PUBLIC_API_ANALYSIS.md`
- View statistics: `PUBLIC_API_STATISTICS.md`
- Examples: `examples/` directory
- Tests: `tests/` directory

