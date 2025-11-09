# OpenCV-Rust Public API - Statistics & Summary

## API Coverage by Module

### Core Module (src/core/)
**Structs: 5**
- Mat, Point, Point2f, Point3f, Size, Rect, Scalar

**Enums: 4**
- MatDepth (3 variants: U8, F32, F64)
- ColorConversionCode (8 variants)
- InterpolationFlag (5 variants)
- ThresholdType (5 variants)

**Functions: 11**
- Matrix arithmetic: add, subtract, multiply
- Bitwise operations: bitwise_and, bitwise_or, bitwise_not
- Channel operations: split, merge
- Statistics: mean, min_max_loc, abs_diff

### Image Processing Module (src/imgproc/)
**Sub-modules: 11**
- color, filter, geometric, threshold, morphology, edge, drawing, contours, histogram, hough, advanced_filter

**Main Functions: 40+**
- Filters (3): gaussian_blur, blur, median_blur
- Color (1): cvt_color
- Geometric (7): resize, flip, rotate, warp_affine, warp_perspective, get_rotation_matrix_2d, get_affine_transform
- Threshold (2): threshold, adaptive_threshold
- Morphology (5): get_structuring_element, erode, dilate, morphology_ex, morph_close
- Edge (4): sobel, canny, laplacian, scharr
- Drawing (8): line, rectangle, circle, circle_filled, ellipse, polylines, fill_poly, put_text
- Contours (6): find_contours, contour_area, arc_length, approx_poly_dp, bounding_rect, moments
- Histogram (4): calc_hist, equalize_hist, compare_hist, calc_back_project
- Advanced Filters (8): bilateral_filter, guided_filter, distance_transform, watershed, gabor_filter, laplacian_of_gaussian, non_local_means_denoising, anisotropic_diffusion

**Enums: 6**
- MorphType, MorphShape, RetrievalMode, ChainApproxMode, AdaptiveThresholdMethod, DistanceType, HistCompMethod, RotateCode

### Feature Detection Module (src/features2d/)
**Detector/Descriptor Structs: 9**
- ORB, AKAZE, BRIEF (2 versions), SimpleSIFT, BRISK, FREAK, KAZE, KeyPoint

**Matcher Structs: 2**
- BFMatcher, DMatch

**Functions: 3**
- harris_corners, good_features_to_track, fast

**Total Methods (approximate): 30+**

### Machine Learning Module (src/ml/)
**Classifier/Regressor Structs: 7**
- SVM, DecisionTree, RandomForest, KNearest, ANN_MLP, KMeans (function), Boost (implemented)

**Total Methods (approximate): 50+**
- SVM: new, train, predict, predict_with_confidence (4)
- DecisionTree: classifier, regressor, with_max_depth, with_min_samples_split, with_min_samples_leaf, train, predict, get_depth, get_leaf_count (9)
- RandomForest: classifier, regressor, with_max_depth, with_min_samples_split, with_max_features, train, predict, predict_proba, n_trees, feature_importances (10)
- KNearest: classifier, regressor, with_algorithm, with_k, train, predict, find_nearest, predict_with_distance (8)
- ANN_MLP: new, set_learning_rate, set_activation_function, train, predict, get_weights, set_weights (7)

### Video Module (src/video/)
**Tracker Structs: 3**
- BackgroundSubtractorMOG2, BackgroundSubtractorKNN, MeanShiftTracker, CamShiftTracker

**Functions/Submodules: 3**
- background_subtraction, tracking, optical_flow

**Total Methods (approximate): 15+**

### Video I/O Module (src/videoio/)
**Structs: 2**
- VideoCapture, VideoWriter

**Functions: 2**
- get_available_codecs, is_codec_available

**Enums: 3**
- FourCC (multiple codec variants)
- VideoCaptureProperty
- ImreadFlag

**Total Methods (approximate): 20+**

### Object Detection Module (src/objdetect/)
**Detector Structs: 4**
- CascadeClassifier, QRCodeDetector, HOGDescriptor, ArucoDictionary

**Total Methods (approximate): 15+**

### Photo Module (src/photo/)
**Sub-modules: 4**
- denoising, hdr, super_resolution, seam_carving

**Function Categories: 15+**
- Denoising: fast_nl_means_denoising_colored, bilateral_filter, anisotropic_diffusion, median_filter, wiener_filter, total_variation_denoise
- Super Resolution: SuperResolutionBicubic, SuperResolutionBP, SuperResolutionExample

### Calibration 3D Module (src/calib3d/)
**Structs: 2**
- CameraMatrix, DistortionCoefficients

**Sub-modules: 5**
- camera, fisheye, homography, pnp, stereo

**Total Functions (approximate): 20+**
- Camera: calibrate_camera, rodrigues
- Homography: find_homography, apply_homography, warp_perspective, decompose_homography
- PnP: solve_pnp
- Stereo: stereo_calibrate, compute_stereo_disparity, triangulate_point, stereo_rectify

### Deep Learning Module (src/dnn/)
**Structs: 8**
- Network, NetworkBuilder, Blob, ConvolutionLayer, PoolingLayer, ActivationLayer, FullyConnectedLayer, FlattenLayer, SoftmaxLayer

**Trait: 1**
- Layer

**Functions: 4**
- read_net_from_caffe, read_net_from_tensorflow, read_net_from_onnx, read_net_from_torch

### FLANN Module (src/flann/)
**Index Structs: 4**
- Index, LinearIndex, KDTree, LSHIndex

**Utility Structs: 3**
- IndexParams, DistanceType, Algorithm, CentersInit

**Total Methods (approximate): 30+**

### Shape Module (src/shape/)
**Functions: 8**
- arc_length, contour_area, circularity, convexity, convex_hull, min_enclosing_circle, bounding_rect, aspect_ratio, extent, solidity

### Stitching Module (src/stitching/)
**Structs: 3**
- PanoramaStitcher, MultiBandBlender, FeatherBlender

**Total Methods (approximate): 10+**

### GPU Module (src/gpu/)
**Functions: 4 (async variants)**
- canny_gpu_async, canny_gpu
- gaussian_blur_gpu_async, gaussian_blur_gpu
- resize_gpu_async, resize_gpu
- threshold_gpu_async, threshold_gpu

**Initialization: 2**
- init_gpu, gpu_available

### WASM Module (src/wasm/)
**Structs: 1**
- WasmMat

**Async Functions: 4**
- gaussian_blur_wasm, resize_wasm, threshold_wasm, canny_wasm

**Initialization: 2**
- wasm_init, init_thread_pool

**Utility: 2**
- is_gpu_available, get_version

---

## Overall API Statistics

| Category | Count |
|----------|-------|
| Top-level Modules | 16 |
| Sub-modules | 25+ |
| Structs | 80+ |
| Enums | 30+ |
| Traits | 1 |
| Module-level Functions | 150+ |
| Struct Methods | 300+ |
| Total Public Methods/Functions | 450+ |

---

## Method Type Distribution

| Method Type | Count | Percentage |
|-----------|-------|-----------|
| Constructors (.new, .from_*) | 40+ | 8% |
| Builders (.with_*) | 50+ | 11% |
| Getters (properties) | 60+ | 13% |
| Setters (.set_*) | 40+ | 9% |
| Operations (core functionality) | 200+ | 45% |
| Queries (.is_*, .contains) | 20+ | 4% |
| Type Conversions (.convert_to, .to_*) | 15+ | 3% |
| Utility Methods | 25+ | 7% |

---

## Naming Convention Breakdown

### Functions using specific verb patterns:

**Detection/Finding (30+)**
- find_*, detect_*, harris_*, good_features_*, fast_*

**Transformation/Processing (60+)**
- *_blur, *_filter, cvt_color, threshold, erode, dilate, resize, flip, rotate, warp_*

**Computation (25+)**
- calc_*, compute, apply, train, predict, forward

**Utility (35+)**
- get_*, set_*, is_*, contains, new, from_*, to_*, clone_*, roi, read, write, release

---

## Enum Naming Patterns

All enums follow PascalCase naming:

**Color-related**: ColorConversionCode, InterpolationFlag
**Type-related**: MatDepth, ThresholdType
**Operation-related**: MorphType, MorphShape, RetrievalMode, ChainApproxMode
**Matching/Distance**: DistanceType, HistCompMethod
**Machine Learning**: SVMType, SVMKernelType, ActivationType, PoolType, LayerType
**Video**: FourCC, VideoCaptureProperty, ImreadFlag

All variants use PascalCase with clear descriptive names:
- Example: `ThresholdType::Binary`, `ThresholdType::BinaryInv`, not `THRESH_BINARY`, `THRESH_BINARY_INV`

---

## API Access Patterns

### Pattern 1: Function-based (stateless operations)
```rust
// Pure functions, no state maintained
cvt_color(&src, &mut dst, ColorConversionCode::RgbToGray)?
gaussian_blur(&src, &mut dst, Size::new(5, 5), 1.5)?
threshold(&src, &mut dst, 128.0, 255.0, ThresholdType::Binary)?
```

### Pattern 2: Struct-based with state (detection algorithms)
```rust
// Create detector instance with configuration
let orb = ORB::new(500);
let (keypoints, descriptors) = orb.detect_and_compute(&image)?;
```

### Pattern 3: Builder pattern (complex configuration)
```rust
let rf = RandomForest::classifier(100)
    .with_max_depth(20)
    .with_min_samples_split(5)
    .with_max_features(MaxFeatures::Sqrt);
```

### Pattern 4: In-place operations (memory efficient)
```rust
// Modify destination matrix in place
let mut dst = Mat::new(...)?;
blur(&src, &mut dst, Size::new(5, 5))?;
```

### Pattern 5: Result-based error handling
```rust
// All operations return Result<T, Error>
let mat = Mat::new(100, 100, 3, MatDepth::U8)?;
let gray = mat.at(row, col)?;
```

---

## Key API Design Decisions

1. **Input/Output Parameter Style**
   - Source images: `&Mat` (immutable reference)
   - Destination/output: `&mut Mat` (mutable reference)
   - Return value: `Result<()>` on success, `Error` on failure

2. **Error Handling**
   - All fallible operations use `Result<T>`
   - Custom error types: `Error::InvalidParameter`, `Error::UnsupportedOperation`
   - No panics in public API

3. **Memory Management**
   - No explicit `.release()` needed for most types
   - `.release()` used for resource management (VideoWriter, VideoCapture)
   - RAII patterns for cleanup

4. **Type Safety**
   - Strong typing with enums instead of constants
   - Compile-time parameter validation where possible
   - Generic type parameters for flexible operations

5. **Performance Considerations**
   - Mutable references for in-place operations
   - GPU acceleration when available (optional feature)
   - Parallel processing support (rayon feature)
   - WASM support with async operations

---

## Comparison Matrix: Current vs OpenCV-Rust Bindings

| Aspect | Current Implementation | Official opencv-rust |
|--------|----------------------|---------------------|
| Function Naming | snake_case ✓ | snake_case ✓ |
| Struct Naming | PascalCase ✓ | PascalCase ✓ |
| Method Naming | snake_case ✓ | snake_case ✓ |
| Error Handling | Custom Result<T> | Result<T> with Error |
| Memory Safety | All safe Rust | All safe Rust ✓ |
| Concurrency | Partial (Rayon) | Limited |
| GPU Support | Optional feature | Limited |
| WASM Support | Full async support | None |
| Pure Rust | Yes ✓ | C++ FFI |
| Builder Pattern | Extensive | Limited |

---

## Migration Path from Current to Official bindings

If migration to official `opencv-rust` bindings is needed, key changes would be:

1. **Naming Changes**
   - Enum variants: `RgbToGray` → `COLOR_RGB2GRAY`
   - Less verbose: Many `.with_*()` patterns removed

2. **Function Signature Changes**
   - Some functions may accept different parameter types
   - Error handling style differs

3. **Module Organization**
   - Flatter namespace hierarchy
   - More "cv::" style prefixes in naming

4. **Advanced Features**
   - Loss of WASM support
   - Different async patterns
   - C++ FFI binding differences

---

## Recommendations for Consistency

1. **Keep existing naming** - Current conventions are Rust-idiomatic
2. **Maintain builder patterns** - Excellent for API ergonomics
3. **Document comparisons** - Create migration guides if needed
4. **Consistent verb usage** - Continue using descriptive action words
5. **Type safety first** - Avoid exposing magic numbers
6. **Error handling** - Maintain Result<T> for all fallible operations

