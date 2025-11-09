# OpenCV-Rust Public API Analysis

## Overview
This analysis covers all public APIs exposed in the current opencv-rust implementation, including method naming conventions, struct implementations, and module organization.

## Repository Structure
- **Main Entry Point**: `src/lib.rs`
- **Prelude Module**: `src/prelude.rs` - re-exports common types
- **Core Modules**:
  - `core/` - Basic data structures (Mat, Point, Size, Rect, Scalar)
  - `imgproc/` - Image processing functions
  - `imgcodecs/` - Image I/O
  - `features2d/` - Feature detection and description
  - `video/` - Video processing
  - `videoio/` - Video I/O
  - `ml/` - Machine Learning
  - `objdetect/` - Object Detection
  - `photo/` - Computational Photography
  - `calib3d/` - Camera Calibration
  - `dnn/` - Deep Neural Networks
  - `flann/` - Fast Library for Approximate Nearest Neighbors
  - `stitching/` - Image Stitching
  - `shape/` - Shape Descriptors
  - `gpu/` - GPU Operations (optional feature)
  - `wasm/` - WebAssembly support

---

## Current Naming Conventions

### 1. Functions (Module-level)
All module-level functions use **snake_case**:
- `gaussian_blur()`, `blur()`, `median_blur()`
- `cvt_color()`, `cvt_color()`
- `imread()`, `imwrite()`
- `threshold()`, `adaptive_threshold()`
- `erode()`, `dilate()`
- `sobel()`, `canny()`, `laplacian()`
- `find_contours()`, `contour_area()`, `arc_length()`
- `harris_corners()`, `good_features_to_track()`, `fast()`

### 2. Structs
All struct names use **PascalCase**:
- **Core**: `Mat`, `Point`, `Point2f`, `Point3f`, `Size`, `Rect`, `Scalar`
- **Features2D**: `ORB`, `BRIEF`, `SimpleSIFT`, `AKAZE`, `FREAK`, `BRISK`, `KeyPoint`, `DMatch`, `BFMatcher`
- **ML**: `SVM`, `DecisionTree`, `RandomForest`, `KNearest`, `ANN_MLP`, `KDTree`, `LSHIndex`
- **Video**: `BackgroundSubtractorMOG2`, `BackgroundSubtractorKNN`, `MeanShiftTracker`, `CamShiftTracker`
- **Detection**: `CascadeClassifier`, `QRCodeDetector`, `HOGDescriptor`
- **Other**: `CameraMatrix`, `DistortionCoefficients`, `Network`, `Blob`, etc.

### 3. Enums
All enum names use **PascalCase**:
- `MatDepth`, `ColorConversionCode`, `InterpolationFlag`, `ThresholdType`
- `RetrievalMode`, `ChainApproxMode`
- `MorphType`, `MorphShape`
- `DistanceType`, `HistCompMethod`
- `SVMType`, `SVMKernelType`
- `FourCC`, `VideoCaptureProperty`
- `ActivationType`, `PoolType`, `LayerType`

Enum variants also use **PascalCase**:
- `MatDepth::U8`, `MatDepth::F32`, `MatDepth::F64`
- `ColorConversionCode::RgbToGray`, `ColorConversionCode::BgrToGray`
- `InterpolationFlag::Linear`, `InterpolationFlag::Cubic`
- `ThresholdType::Binary`, `ThresholdType::BinaryInv`

### 4. Struct Methods
All methods on structs use **snake_case**:

#### Constructor Methods:
- `.new()` - Basic constructor
- `.new_with_default()` - Constructor with default values
- `.from_*()` - Conversion constructors (e.g., `from_image_data()`, `from_matrix()`)
- `.from_raw()` - Create from raw data

#### Builder Pattern Methods:
- `.with_*()` - Configuration (e.g., `.with_threshold()`, `.with_scale_factor()`, `.with_params()`)

#### Getter Methods:
- `.rows()`, `.cols()` - Dimensions
- `.channels()`, `.depth()` - Data properties
- `.data()`, `.data_mut()` - Access raw data
- `.width()`, `.height()` - Dimensions for other types
- `.size()` - Overall size
- `.area()` - Calculated property
- `.is_opened()`, `.is_empty()` - Boolean queries
- `.contains()` - Membership test

#### Setter Methods:
- `.set_*()` - Mutators (e.g., `.set_to()`, `.set_f32()`, `.set_u16()`)
- `.at()`, `.at_mut()` - Element access

#### Operation Methods:
- `.detect_and_compute()` - Detection and computation
- `.predict()` - Prediction/inference
- `.train()` - Training
- `.apply()` - Apply operation
- `.compute()` - Computation
- `.match_descriptors()` - Matching
- `.knn_match()`, `.knn_search()` - K-nearest neighbors
- `.read()` - Read frames/data
- `.write()` - Write data
- `.forward()` - Network forward pass
- `.blend()` - Image blending
- `.stitch()` - Image stitching
- `.clone_mat()` - Clone operation
- `.roi()` - Region of interest
- `.release()` - Resource cleanup
- `.get_background_image()` - Get background

#### Type Conversion Methods:
- `.convert_to()` - Type/depth conversion
- `.from_chars()`, `.from_str()`, `.from_int()` - Conversion constructors
- `.to_int()` - Conversion to int

---

## Module Organization

### Core Module (`src/core/`)
**File Structure:**
```
core/
├── mod.rs        (re-exports: Mat, MatDepth, and operations)
├── mat.rs        (Mat struct definition)
├── mat_typed.rs  (Typed accessors for Mat: at_f32, set_f64, etc.)
├── types.rs      (Point, Point2f, Point3f, Size, Rect, Scalar, Enums)
└── operations.rs (Matrix operations: add, subtract, multiply, etc.)
```

**Public API Surface:**
- Structs: `Mat`, `Point`, `Point2f`, `Point3f`, `Size`, `Rect`, `Scalar`
- Enums: `MatDepth`, `ColorConversionCode`, `InterpolationFlag`, `ThresholdType`
- Functions: `add()`, `subtract()`, `multiply()`, `bitwise_and()`, `split()`, `merge()`, `mean()`, etc.

### Image Processing Module (`src/imgproc/`)
**File Structure:**
```
imgproc/
├── mod.rs             (re-exports all submodules)
├── color.rs           (cvt_color)
├── filter.rs          (gaussian_blur, blur, median_blur)
├── geometric.rs       (resize, flip, warp_affine, rotate, etc.)
├── threshold.rs       (threshold, adaptive_threshold)
├── morphology.rs      (erode, dilate, etc.)
├── edge.rs            (sobel, canny, laplacian, scharr)
├── drawing.rs         (line, rectangle, circle, polylines, text)
├── contours.rs        (find_contours, contour operations)
├── histogram.rs       (histogram operations)
├── hough.rs           (Hough transforms)
└── advanced_filter.rs (bilateral_filter, guided_filter, etc.)
```

### Features2D Module (`src/features2d/`)
**File Structure:**
```
features2d/
├── mod.rs        (re-exports)
├── keypoints.rs  (KeyPoint struct, harris_corners, good_features_to_track, fast)
├── descriptors.rs (ORB, BRIEF, SimpleSIFT)
├── akaze.rs      (AKAZE detector)
├── brisk.rs      (BRISK detector)
├── orb.rs        (ORB detector)
├── brief.rs      (BRIEF descriptor)
├── freak.rs      (FREAK descriptor)
├── kaze.rs       (KAZE detector)
├── matching.rs   (DMatch, BFMatcher, matching functions)
└── sift_f32.rs   (SIFT implementation)
```

### ML Module (`src/ml/`)
**File Structure:**
```
ml/
├── mod.rs              (re-exports)
├── svm.rs              (SVM: Support Vector Machine)
├── dtree.rs            (DecisionTree classifier/regressor)
├── random_forest.rs    (RandomForest classifier/regressor)
├── knearest.rs         (KNearest classifier/regressor)
├── ann.rs              (ANN_MLP: Artificial Neural Network)
├── kmeans.rs           (KMeans clustering)
└── boost.rs            (Boosting algorithms)
```

### Video Module (`src/video/`)
**File Structure:**
```
video/
├── mod.rs                      (re-exports)
├── tracking.rs                 (BackgroundSubtractorMOG2, MeanShiftTracker, CamShiftTracker)
├── background_subtraction.rs   (BackgroundSubtractorMOG2, BackgroundSubtractorKNN)
├── optical_flow.rs             (Optical flow algorithms)
├── camshift.rs                 (CamShift tracking)
├── advanced_tracking.rs        (Advanced tracking methods)
└── mod.rs
```

---

## Detailed Public API Listing

### Core Structs and Their Methods

#### Mat
- Constructor: `.new(rows, cols, channels, depth)`, `.new_with_default()`, `.from_raw()`
- Dimensions: `.rows()`, `.cols()`, `.channels()`, `.size()`, `.depth()`
- Data access: `.data()`, `.data_mut()`, `.at()`, `.at_mut()`
- Operations: `.clone_mat()`, `.roi()`, `.set_to()`
- Type-specific: `.at_f32()`, `.set_f32()`, `.at_f64()`, `.set_f64()`, `.at_u16()`, `.set_u16()`, `.convert_to()`
- Query: `.is_empty()`

#### Point / Point2f / Point3f
- Constructor: `.new(x, y)` / `.new(x, y, z)`
- Properties: `.x`, `.y`, `.z`

#### Size
- Constructor: `.new(width, height)`
- Methods: `.area()`
- Properties: `.width`, `.height`

#### Rect
- Constructor: `.new(x, y, width, height)`
- Methods: `.area()`, `.top_left()`, `.bottom_right()`, `.contains(point)`
- Properties: `.x`, `.y`, `.width`, `.height`

#### Scalar
- Constructors: `.new(v0, v1, v2, v3)`, `.all(v)`, `.from_rgb()`, `.from_rgba()`
- Properties: `.val[]`
- Operators: `+`, `-`, `*`, `/`

### Image Processing Functions

#### Filters
- `gaussian_blur(src, dst, ksize, sigma)`
- `blur(src, dst, ksize)`
- `median_blur(src, dst, ksize)`

#### Color Operations
- `cvt_color(src, dst, code)`

#### Geometric
- `resize(src, dst, dsize, interpolation)`
- `flip(src, dst, flip_code)`
- `rotate(src, dst, rotate_code)`
- `warp_affine(src, dst, M, dsize)`
- `warp_perspective(src, dst, M, dsize)`
- `get_rotation_matrix_2d(center, angle, scale)`
- `get_affine_transform(src, dst)`

#### Edge Detection
- `sobel(src, dst, dx, dy)`
- `canny(src, dst, threshold1, threshold2)`
- `laplacian(src, dst, ksize)`
- `scharr(src, dst, dx, dy)`

#### Thresholding
- `threshold(src, dst, thresh, maxval, type)`
- `adaptive_threshold(src, dst, maxval, method, block_size, c)`

#### Morphology
- `get_structuring_element(shape, ksize)`
- `erode(src, dst, kernel)`
- `dilate(src, dst, kernel)`
- `morphology_ex(src, dst, op, kernel)`

#### Drawing
- `line(img, pt1, pt2, color, thickness)`
- `rectangle(img, rect, color, thickness)`
- `circle(img, center, radius, color)`
- `circle_filled(img, center, radius, color)`
- `ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)`
- `polylines(img, pts, isClosed, color, thickness)`
- `fill_poly(img, pts, color)`
- `put_text(img, text, org, fontFace, fontScale, color, thickness)`

#### Contours
- `find_contours(image, mode, method)` -> Vec<Contour>
- `contour_area(contour)`
- `arc_length(contour, closed)`
- `approx_poly_dp(contour, epsilon, closed)`
- `bounding_rect(contour)`
- `moments(contour)`

#### Histogram
- `calc_hist(images, channels, mask, hist_size, ranges)`
- `equalize_hist(src, dst)`
- `compare_hist(h1, h2, method)`
- `calc_back_project(images, channels, hist, ranges)`

---

## Machine Learning APIs

### SVM
```rust
pub struct SVM { ... }
- new(svm_type, kernel_type)
- train(samples, labels)
- predict(sample) -> f64
- predict_with_confidence(sample) -> (f64, f64)
```

### KNearest
```rust
pub struct KNearest { ... }
- classifier(k) / regressor(k)
- with_algorithm(algo)
- with_k(k)
- train(data, labels)
- predict(sample) -> f64
- find_nearest(sample, k) -> Vec<(usize, f64)>
- predict_with_distance(sample, distance_fn)
```

### RandomForest
```rust
pub struct RandomForest { ... }
- classifier(n_trees) / regressor(n_trees)
- with_max_depth(depth)
- with_min_samples_split(samples)
- with_max_features(features)
- train(data, labels)
- predict(sample) -> f64
- predict_proba(sample) -> HashMap<i32, f64>
- n_trees() -> usize
- feature_importances(n_features) -> Vec<f64>
```

### DecisionTree
```rust
pub struct DecisionTree { ... }
- classifier() / regressor()
- with_max_depth(depth)
- with_min_samples_split(samples)
- with_min_samples_leaf(samples)
- train(data, labels)
- predict(sample) -> f64
- get_depth() -> usize
- get_leaf_count() -> usize
```

### ANN_MLP (Artificial Neural Network)
```rust
pub struct ANN_MLP { ... }
- new(layer_sizes)
- set_learning_rate(rate)
- set_activation_function(activation)
- train(data, labels)
- predict(input) -> Vec<f64>
- get_weights() -> Vec<Vec<Vec<f64>>>
- set_weights(weights)
```

### KMeans
```rust
pub fn kmeans(data, k, max_iterations, epsilon) -> (Vec<Vec<f64>>, Vec<i32>)
```

---

## Feature Detection APIs

### ORB
```rust
pub struct ORB { ... }
- new(n_features)
- detect_and_compute(image) -> (Vec<KeyPoint>, Vec<Vec<u8>>)
```

### AKAZE
```rust
pub struct AKAZE { ... }
- new()
- with_threshold(threshold)
- detect_and_compute(image) -> (Vec<KeyPoint>, Vec<Vec<u8>>)
```

### BRIEF
```rust
pub struct BRIEF { ... }
- new()
- with_params(bytes, patch_size, use_orientation)
- compute(image, keypoints) -> Vec<Vec<u8>>
```

### BFMatcher
```rust
pub struct BFMatcher { ... }
- new(distance_type, cross_check)
- match_descriptors(desc1, desc2) -> Vec<DMatch>
- knn_match(desc1, desc2, k) -> Vec<Vec<DMatch>>
- radius_match(desc1, desc2, radius) -> Vec<Vec<DMatch>>
```

---

## Video I/O APIs

### VideoCapture
```rust
pub struct VideoCapture { ... }
- from_file(path)
- from_camera(device_id)
- is_opened() -> bool
- read(frame) -> Result<bool>
- get(prop) -> f64
- set(prop, value)
- release()
- get_backend_name() -> &str
```

### VideoWriter
```rust
pub struct VideoWriter { ... }
- new(filename, fourcc, fps, frame_size)
- is_opened() -> bool
- write(frame)
- release()
- frame_count() -> usize
- get_fps() -> f64
- get_frame_size() -> (usize, usize)
- get_fourcc() -> FourCC
```

---

## Comparison with Official OpenCV-Rust Bindings

The official `opencv` crate (twistedfall/opencv-rust) uses:

### Naming Similarities ✓
1. snake_case for functions: `gaussian_blur()`, `cvt_color()`
2. PascalCase for structs: `Mat`, `Point`, `Size`
3. PascalCase for enums and variants
4. snake_case for methods: `.rows()`, `.cols()`, `.clone()`, etc.
5. Builder pattern: `.with_*()` methods

### Potential Differences
1. **Official OpenCV-Rust** may use:
   - More abbreviated method names in some cases
   - Different enum variant names (e.g., `COLOR_RGB2GRAY` vs `ColorConversionCode::RgbToGray`)
   - Different struct organization (namespace hierarchies)
   - More C++-like naming for advanced features

2. **This Implementation** uses:
   - Verbose enum variants for clarity (`RgbToGray` instead of `RGB2GRAY`)
   - Flatter module structure
   - Builder patterns extensively
   - Pure Rust implementations without C++ bindings

---

## Summary of Naming Patterns

| Item | Current Convention | Example |
|------|-------------------|---------|
| Functions | snake_case | `gaussian_blur()`, `find_contours()` |
| Structs | PascalCase | `Mat`, `ORB`, `SVM` |
| Enums | PascalCase | `MatDepth`, `ColorConversionCode` |
| Enum Variants | PascalCase | `MatDepth::U8`, `ThresholdType::Binary` |
| Methods | snake_case | `.rows()`, `.detect_and_compute()` |
| Constructors | `.new()`, `.from_*()` | `.new()`, `.from_matrix()` |
| Builders | `.with_*()` | `.with_threshold()`, `.with_scale_factor()` |
| Getters | direct name | `.rows()`, `.channels()` |
| Setters | `.set_*()` | `.set_to()`, `.set_f32()` |
| Boolean Methods | `.is_*()` | `.is_opened()`, `.is_empty()` |
| Module Exports | `pub use` re-exports | All submodules re-exported to parent |

---

## Key Design Observations

1. **Consistent snake_case** for all functions and methods
2. **Clear separation** between constructors, builders, getters, and operations
3. **Enum variants** use descriptive names rather than cryptic abbreviations
4. **Module re-exports** flatten the API surface for convenience
5. **Result<> return type** for fallible operations
6. **Owned parameters** for efficiency (avoid unnecessary copying)
7. **In-place mutations** with `&mut` for output parameters
8. **Builder pattern** for complex object construction
9. **Type-safe enums** instead of magic numbers
10. **Generic lifetime management** (mostly implicit via references)

