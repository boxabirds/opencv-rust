# OpenCV-Rust Public API Documentation Index

This document serves as a guide to the comprehensive API documentation generated for the opencv-rust project.

## Documentation Files

### 1. **API Quick Reference** (Start Here!)
**File**: `/home/user/opencv-rust/API_QUICK_REFERENCE.md`

Quick lookup guide with:
- Essential data types and patterns
- Most common APIs by category
- Pattern reference (constructor, builder, in-place operations)
- Module structure overview
- Performance tips and troubleshooting

**Best for**: Quick lookups, common usage patterns, quick reference

---

### 2. **Complete API Analysis**
**File**: `/home/user/opencv-rust/PUBLIC_API_ANALYSIS.md`

Comprehensive analysis covering:
- Full module structure breakdown
- Detailed public API listing by module
- Complete naming convention documentation
- Method type categorization
- Comparison with official opencv-rust bindings

**Best for**: Understanding the complete API surface, naming patterns, detailed design decisions

---

### 3. **API Statistics & Details**
**File**: `/home/user/opencv-rust/PUBLIC_API_STATISTICS.md`

Statistical breakdown including:
- API coverage by module with counts
- Overall statistics (80+ structs, 30+ enums, 450+ methods)
- Method type distribution
- Naming convention breakdown by category
- API access patterns
- Key design decisions
- Migration path analysis

**Best for**: Understanding scope of implementation, API patterns, design decisions

---

## Key Findings Summary

### Naming Conventions (Consistent Rust Style)

| Category | Convention | Examples |
|----------|-----------|----------|
| Functions | `snake_case` | `gaussian_blur()`, `find_contours()` |
| Structs | `PascalCase` | `Mat`, `ORB`, `SVM`, `RandomForest` |
| Methods | `snake_case` | `.rows()`, `.detect_and_compute()`, `.predict()` |
| Enums | `PascalCase` | `MatDepth`, `ColorConversionCode`, `ThresholdType` |
| Enum Variants | `PascalCase` | `MatDepth::U8`, `ThresholdType::Binary` |
| Builder Methods | `.with_*()` | `.with_threshold()`, `.with_scale_factor()` |
| Getters | Direct name | `.rows()`, `.channels()` |
| Setters | `.set_*()` | `.set_to()`, `.set_f32()` |
| Boolean Methods | `.is_*()` | `.is_opened()`, `.is_empty()` |

### API Coverage Overview

**450+ Public Methods/Functions across:**
- **16** top-level modules
- **80+** structs
- **30+** enums
- **150+** module-level functions
- **300+** struct methods

### Module Organization

```
Core Modules (Stable):
├── core/              (Mat, Point, Size, Rect, operations)
├── imgproc/           (11 sub-modules, 40+ functions)
├── features2d/        (9 detectors, matching, 30+ methods)
├── ml/                (7 classifiers, 50+ methods)
├── video/             (tracking, background subtraction)
├── videoio/           (VideoCapture, VideoWriter)
├── imgcodecs/         (imread, imwrite)

Advanced Modules:
├── objdetect/         (detection algorithms)
├── photo/             (denoising, super-resolution)
├── calib3d/           (camera calibration, stereo)
├── dnn/               (neural networks)
├── flann/             (nearest neighbor search)
├── stitching/         (image stitching, panorama)
├── shape/             (contour analysis)

Optional Modules:
├── gpu/               (GPU acceleration, optional feature)
└── wasm/              (WebAssembly support, optional)
```

---

## Key Design Patterns Used

### 1. Function-Based (Stateless)
```rust
// Pure functions for simple operations
cvt_color(&src, &mut dst, ColorConversionCode::RgbToGray)?;
gaussian_blur(&src, &mut dst, Size::new(5, 5), 1.5)?;
```

### 2. Struct-Based with State
```rust
// Detectors/classifiers maintain state
let orb = ORB::new(500);
let (keypoints, descriptors) = orb.detect_and_compute(&image)?;
```

### 3. Builder Pattern
```rust
// Complex configuration
let rf = RandomForest::classifier(100)
    .with_max_depth(20)
    .with_min_samples_split(5);
```

### 4. In-Place Operations
```rust
// Memory efficient
let mut dst = Mat::new(...)?;
blur(&src, &mut dst, kernel)?;
```

### 5. Result-Based Error Handling
```rust
// All operations return Result<T>
let result = operation(&input)?;  // Using ? operator
```

---

## Comparison with Official opencv-rust

### Similarities (Same Conventions)
- snake_case for functions
- PascalCase for structs and enums
- snake_case for methods
- Result<T> for error handling

### Key Differences
| Aspect | Current | Official |
|--------|---------|----------|
| Implementation | Pure Rust | C++ FFI |
| Enum Variants | Descriptive (`RgbToGray`) | Cryptic (`COLOR_RGB2GRAY`) |
| Builder Pattern | Extensive | Limited |
| GPU Support | Full (async) | Limited |
| WASM Support | Full (async) | None |
| Concurrency | Rayon support | Limited |

---

## Quick Links to Module Documentation

### Image Processing
- **See**: `PUBLIC_API_ANALYSIS.md` → "Image Processing Module"
- **Functions**: 40+ including filters, geometric ops, edge detection, thresholding
- **Notable**: gaussian_blur, canny, find_contours, threshold

### Machine Learning
- **See**: `PUBLIC_API_ANALYSIS.md` → "Detailed Public API Listing → Machine Learning APIs"
- **Classifiers**: SVM, RandomForest, DecisionTree, KNearest, ANN
- **Clustering**: KMeans

### Feature Detection
- **See**: `PUBLIC_API_ANALYSIS.md` → "Detailed Public API Listing → Feature Detection APIs"
- **Detectors**: ORB, AKAZE, BRIEF, BRISK, FREAK
- **Matching**: BFMatcher, DMatch

### Video I/O
- **See**: `PUBLIC_API_ANALYSIS.md` → "Detailed Public API Listing → Video I/O APIs"
- **Capture**: VideoCapture (files, cameras)
- **Writing**: VideoWriter (codecs, formats)

---

## File Locations (Absolute Paths)

### Documentation Files
- `/home/user/opencv-rust/API_QUICK_REFERENCE.md` (Quick lookup)
- `/home/user/opencv-rust/PUBLIC_API_ANALYSIS.md` (Complete analysis)
- `/home/user/opencv-rust/PUBLIC_API_STATISTICS.md` (Statistics)
- `/home/user/opencv-rust/API_DOCUMENTATION_INDEX.md` (This file)

### Source Code Organization
- `/home/user/opencv-rust/src/lib.rs` (Entry point)
- `/home/user/opencv-rust/src/core/` (Core types and operations)
- `/home/user/opencv-rust/src/imgproc/` (Image processing)
- `/home/user/opencv-rust/src/features2d/` (Feature detection)
- `/home/user/opencv-rust/src/ml/` (Machine learning)
- `/home/user/opencv-rust/src/video/` (Video processing)
- `/home/user/opencv-rust/src/videoio/` (Video I/O)
- `/home/user/opencv-rust/src/prelude.rs` (Common re-exports)

### Examples
- `/home/user/opencv-rust/examples/basic_operations.rs`
- `/home/user/opencv-rust/examples/image_processing.rs`
- `/home/user/opencv-rust/examples/comprehensive_demo.rs`

### Tests
- `/home/user/opencv-rust/tests/test_core.rs`
- `/home/user/opencv-rust/tests/test_imgproc.rs`
- `/home/user/opencv-rust/tests/test_ml.rs`
- And many more accuracy/feature tests...

---

## How to Use These Documents

### For API Users
1. Start with `API_QUICK_REFERENCE.md` for quick lookups
2. Refer to `PUBLIC_API_ANALYSIS.md` for detailed method signatures
3. Check examples in `/home/user/opencv-rust/examples/` for usage patterns

### For API Designers
1. Review `PUBLIC_API_STATISTICS.md` for design patterns
2. Check `PUBLIC_API_ANALYSIS.md` for naming conventions
3. Use comparison section for migration considerations

### For Contributors
1. Review naming conventions in `API_QUICK_REFERENCE.md`
2. Follow patterns documented in `PUBLIC_API_STATISTICS.md`
3. Keep consistency with existing module organization

---

## Method Naming Quick Reference

### Constructor Methods
- `.new()` - Basic constructor
- `.new_with_default()` - With default values
- `.from_*()` - Conversion constructors
- `.classifier()` / `.regressor()` - ML classifiers

### Configuration Methods
- `.with_*()` - Builder pattern configuration
- `.set_*()` - Direct setters

### Query Methods
- `.rows()`, `.cols()` - Dimensions
- `.channels()`, `.depth()` - Properties
- `.is_*()` - Boolean queries
- `.contains()` - Membership tests
- `.area()` - Calculated properties

### Operation Methods
- `.detect_and_compute()` - Feature detection
- `.predict()` - Inference/prediction
- `.train()` - Training
- `.apply()` - Apply operation
- `.match_descriptors()` - Descriptor matching
- `.read()` / `.write()` - I/O operations
- `.forward()` - Network inference

### Data Access
- `.data()` / `.data_mut()` - Raw data access
- `.at()` / `.at_mut()` - Element access
- `.at_f32()`, `.set_f64()` - Type-specific access

---

## Standards and Conventions Summary

### What's Consistent:
- All functions use snake_case
- All struct/enum names use PascalCase
- All methods use snake_case
- Builder pattern uses `.with_*()` consistently
- Error handling always returns `Result<T>`
- Input parameters use `&Mat` or `&[T]`
- Output parameters use `&mut Mat` or `&mut T`

### What's Idiomatic Rust:
- No unnecessary abbreviations
- Clear, descriptive names
- Standard Rust naming conventions
- Standard error handling patterns
- Standard builder patterns

---

## Notes on Comparisons

The current implementation follows **standard Rust conventions** more closely than the official opencv-rust bindings, which are closer to C++ OpenCV naming. The key advantage is:

- **More readable**: `ColorConversionCode::RgbToGray` vs `COLOR_RGB2GRAY`
- **More idiomatic**: Uses Rust patterns (builders, Result<>, etc.)
- **Better safety**: All safe Rust, no C++ FFI
- **Better web support**: Full WASM support with async operations

---

## Generation Information

- **Generated**: 2025-11-09
- **Coverage**: All public APIs in `src/` directory
- **Total Items Analyzed**: 450+ public methods/functions
- **Module Coverage**: 16 top-level modules + 25+ sub-modules
- **Naming Consistency**: 100% following Rust conventions

---

**For questions about specific APIs, start with the Quick Reference and then consult the Complete Analysis.**

