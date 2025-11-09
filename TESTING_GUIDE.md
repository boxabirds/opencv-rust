# Testing Guide - Bit-Level Accuracy Validation

This guide explains how to verify bit-level accuracy of the optimized OpenCV implementations.

## Overview

The codebase includes comprehensive testing infrastructure to ensure optimizations don't change results:

- **335 total tests** (212 unit + 123 integration from OpenCV)
- **100% pass rate** maintained throughout optimization
- **Bit-level accuracy validation** tools for detailed verification
- **Deterministic output** verification
- **Visual inspection** capabilities for debugging

---

## Running Tests

### Run All Tests

```bash
# Run all tests
cargo test

# Run with output (shows assertions)
cargo test -- --nocapture
```

### Run Specific Test Categories

```bash
# Run only accuracy tests
cargo test --test test_accuracy_canny

# Run only integration tests (from OpenCV test suite)
cargo test --test test_imgproc

# Run specific test by name
cargo test test_canny_deterministic
```

### Visual Inspection Tests

Some tests are marked `#[ignore]` and only run with `--ignored` flag. These print visual output:

```bash
# Run visual inspection test for Canny
cargo test --test test_accuracy_canny test_canny_visual_inspection -- --ignored --nocapture
```

**Example output:**
```
Input (20x20, 1 channels):
  Row 0:   0   0   0   0 ...
  Row 1:   0   0   0   0 ...

Canny output:
  Row 0:   0   0   0   0 ...
  Row 1:   0 255   0 255 ...
```

---

## Test Utilities

### Available Test Utilities (in `tests/test_utils.rs`)

#### 1. `assert_images_equal(actual, expected, test_name)`

Verifies two images are **bit-exact identical**:

```rust
use test_utils::assert_images_equal;

let img1 = Mat::new_with_default(10, 10, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();
let img2 = process_image(&img1);

// Fails if ANY pixel differs
assert_images_equal(&img1, &img2, "test_processing");
```

**Output on failure:**
```
test_processing: Pixel differences found!
Total pixels: 100
Differing pixels: 5 (5.00%)
Max difference: 3
First diff at (5, 5) ch0: actual=131, expected=128, diff=3
  (5, 5) ch0: 131 vs 128 (diff: 3)
  (5, 6) ch0: 130 vs 128 (diff: 2)
  ...
```

#### 2. `assert_images_near(actual, expected, max_diff, test_name)`

Allows tolerance for floating-point rounding:

```rust
use test_utils::assert_images_near;

// Allow up to 1 pixel difference due to rounding
assert_images_near(&result, &expected, 1, "test_with_tolerance");
```

#### 3. `compute_diff_stats(actual, expected)`

Get detailed statistics about differences:

```rust
use test_utils::compute_diff_stats;

let stats = compute_diff_stats(&img1, &img2);
println!("{}", stats);
// Output: DiffStats { total: 1000, different: 50 (5.000%), max: 3, mean: 1.20 }
```

#### 4. `print_image_data(img, name, max_rows, max_cols)`

Print pixel values for debugging:

```rust
use test_utils::print_image_data;

print_image_data(&result, "Canny Output", 10, 10);
```

---

## Accuracy Test Examples

### Test 1: Deterministic Output

Verifies same input always produces identical output:

```rust
#[test]
fn test_canny_deterministic() {
    let src = create_test_image();

    let mut edges1 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut edges2 = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    canny(&src, &mut edges1, 50.0, 150.0).unwrap();
    canny(&src, &mut edges2, 50.0, 150.0).unwrap();

    // Results should be bit-exact identical
    assert_images_equal(&edges1, &edges2, "Canny should be deterministic");
}
```

### Test 2: Binary Output Validation

Ensures output only contains expected values:

```rust
#[test]
fn test_canny_output_is_binary() {
    let src = create_test_image();
    let mut edges = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    canny(&src, &mut edges, 50.0, 150.0).unwrap();

    // All pixels should be either 0 or 255
    for row in 0..edges.rows() {
        for col in 0..edges.cols() {
            let pixel = edges.at(row, col).unwrap()[0];
            assert!(pixel == 0 || pixel == 255,
                "Pixel at ({}, {}) = {}, expected 0 or 255", row, col, pixel);
        }
    }
}
```

### Test 3: Known Reference Output

Validate against known good results:

```rust
#[test]
fn test_gaussian_blur_known_output() {
    let mut src = Mat::new(5, 5, 1, MatDepth::U8).unwrap();

    // Create known input
    for i in 0..25 {
        src.data_mut()[i] = if i == 12 { 255 } else { 0 }; // Center pixel bright
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    gaussian_blur(&src, &mut dst, Size::new(3, 3), 1.0).unwrap();

    // Verify center pixel is blurred correctly
    let center = dst.at(2, 2).unwrap()[0];
    assert!(center > 100 && center < 200, "Expected center ~150, got {}", center);
}
```

---

## Creating New Accuracy Tests

### Template for New Test

```rust
mod test_utils;
use test_utils::*;

#[test]
fn test_my_operation_accuracy() {
    // 1. Create controlled input
    let src = Mat::new_with_default(10, 10, 1, MatDepth::U8, Scalar::all(128.0)).unwrap();

    // 2. Run operation
    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    my_operation(&src, &mut dst).unwrap();

    // 3. Validate output
    // Option A: Bit-exact comparison
    let expected = create_expected_output();
    assert_images_equal(&dst, &expected, "my_operation");

    // Option B: Range check
    for row in 0..dst.rows() {
        for col in 0..dst.cols() {
            let val = dst.at(row, col).unwrap()[0];
            assert!(val >= min && val <= max, "Pixel ({},{}) out of range", row, col);
        }
    }

    // Option C: Statistical validation
    let stats = compute_diff_stats(&dst, &expected);
    assert!(stats.max_diff <= 2, "Max diff too large: {}", stats.max_diff);
}
```

---

## Regression Testing Strategy

### 1. Create Baseline

When adding a new optimized function:

```rust
#[test]
fn test_optimized_vs_reference() {
    let src = create_test_image();

    // Run both versions
    let mut result_optimized = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    let mut result_reference = Mat::new(1, 1, 1, MatDepth::U8).unwrap();

    optimized_version(&src, &mut result_optimized).unwrap();
    reference_version(&src, &mut result_reference).unwrap();

    // Should produce identical results
    let stats = compute_diff_stats(&result_optimized, &result_reference);
    println!("Diff stats: {}", stats);
    assert_images_equal(&result_optimized, &result_reference, "optimized matches reference");
}
```

### 2. Validate on Multiple Inputs

```rust
#[test]
fn test_operation_on_various_inputs() {
    let test_cases = vec![
        ("uniform", Mat::new_with_default(10, 10, 1, MatDepth::U8, Scalar::all(128.0)).unwrap()),
        ("gradient", create_gradient_image()),
        ("checkerboard", create_checkerboard()),
        ("random", create_random_image()),
    ];

    for (name, src) in test_cases {
        let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
        my_operation(&src, &mut dst).unwrap();

        // Add validation specific to this test case
        validate_output(&dst, name);
    }
}
```

---

## Performance vs Accuracy Trade-offs

### Acceptable Differences

Some optimizations may introduce **minor** differences due to:
- Floating-point rounding order changes
- Parallel processing non-determinism (if any)

**Guidelines:**
- ✅ Acceptable: ±1 pixel value difference due to rounding
- ✅ Acceptable: Different order of equivalent results (e.g., keypoint order)
- ❌ Not acceptable: Structural differences in output
- ❌ Not acceptable: Missing or extra features detected

### Example: Allowing Rounding Differences

```rust
// Use tolerance for operations with floating-point math
assert_images_near(&result, &expected, 1, "gaussian_blur_with_rounding");

// Or validate statistics
let stats = compute_diff_stats(&result, &expected);
assert!(stats.max_diff <= 1, "Max rounding error: {}", stats.max_diff);
assert!(stats.percent_different < 5.0, "Too many pixels differ: {:.2}%", stats.percent_different);
```

---

## Current Test Coverage

### Operations with Comprehensive Bit-Level Accuracy Tests

- ✅ **Canny Edge Detection**: 12 accuracy tests (`tests/test_accuracy_canny.rs`)
  - Determinism verified
  - Binary output verified
  - Edge orientation tests (vertical, horizontal, diagonal)
  - Threshold sensitivity tests
  - Boundary condition tests

- ✅ **Gaussian Blur**: 13 accuracy tests (`tests/test_accuracy_gaussian.rs`)
  - Determinism verified
  - Uniform image handling
  - Edge smoothing validation
  - Kernel size effects (3x3, 5x5, 7x7, 11x11)
  - Sigma parameter effects
  - Energy conservation
  - Multi-channel independence

- ✅ **Threshold Operations**: 12 accuracy tests (`tests/test_accuracy_threshold.rs`)
  - All 5 threshold types (Binary, BinaryInv, Trunc, ToZero, ToZeroInv)
  - Determinism verified
  - Exact boundary conditions (> not ≥)
  - Multi-channel independence
  - Edge case validation

- ✅ **Resize Operations**: 16 accuracy tests (`tests/test_accuracy_resize.rs`)
  - Determinism verified (upscale and downscale)
  - Nearest neighbor pixel-exact validation
  - Bilinear interpolation smoothness
  - Multi-channel independence
  - Value range validation (no overflow)
  - Edge cases (1x1 resize)

- ✅ **FAST Feature Detection**: 14 accuracy tests (`tests/test_accuracy_fast.rs`)
  - Determinism verified
  - Uniform image handling (zero keypoints)
  - Threshold sensitivity
  - NMS effectiveness
  - Border respect validation
  - Keypoint bounds checking

### Operations with Integration Tests (from OpenCV)

- ✅ Gaussian Blur
- ✅ Resize (bilinear, nearest)
- ✅ Threshold operations
- ✅ Sobel derivatives
- ✅ FAST feature detection
- ✅ Harris corners
- ✅ And 117 more from OpenCV test suite

---

## Debugging Failed Tests

### 1. Get Detailed Diff

```rust
// Add this to failing test
let stats = compute_diff_stats(&actual, &expected);
println!("Diff stats: {}", stats);
print_image_data(&actual, "Actual", 20, 20);
print_image_data(&expected, "Expected", 20, 20);
```

### 2. Visual Inspection

```rust
#[test]
#[ignore]
fn debug_test() {
    let result = my_operation(&input);
    print_image_data(&input, "Input", 30, 30);
    print_image_data(&result, "Output", 30, 30);
}
```

Run with: `cargo test debug_test -- --ignored --nocapture`

### 3. Isolate the Issue

```rust
// Test on minimal case
let src = Mat::new(3, 3, 1, MatDepth::U8).unwrap();
// ... set specific pixel values
// ... verify exact output
```

---

## Continuous Integration

### Running Tests in CI

```bash
# Run all tests with output
cargo test -- --nocapture 2>&1 | tee test_output.log

# Run with timing
cargo test -- --test-threads=1 --nocapture

# Check for determinism (run twice, compare outputs)
cargo test > run1.log
cargo test > run2.log
diff run1.log run2.log
```

---

## Summary

✅ **402 total tests** ensure correctness (335 original + 67 new accuracy tests)
  - **67 bit-level accuracy tests** for optimized operations
  - **212 unit tests**
  - **123 integration tests** from OpenCV test suite

✅ **Bit-level accuracy** validation across 5 critical operations
✅ **Deterministic output** verified for all optimized operations
✅ **Visual debugging** tools for manual inspection
✅ **100% pass rate** maintained throughout optimization

**Key Principle**: Optimizations should **never** change output behavior. If they do, the optimization is incorrect.

**See Also**: `BIT_LEVEL_ACCURACY_REPORT.md` for comprehensive accuracy and performance validation results.
