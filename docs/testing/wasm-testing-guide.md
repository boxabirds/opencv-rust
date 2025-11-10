# WASM Testing Guide: OpenCV.js API Parity

**Date**: 2025-11-10
**Status**: ✅ Infrastructure Complete
**Priority**: 2 (from docs/plan.md)

---

## Overview

This guide establishes the testing approach for ensuring 100% OpenCV.js API parity across all 141 WASM operations. The testing infrastructure enables safe refactoring and validates the claim: "Our WASM bindings are 100% compatible with OpenCV.js."

---

## Goals

1. **API Parity**: Verify 100% signature compatibility with OpenCV.js
2. **Behavioral Compatibility**: Ensure operations produce equivalent results
3. **Backend Validation**: Verify CPU and GPU backends produce consistent results
4. **Refactoring Safety**: Enable confident code reorganization with test coverage
5. **Performance Validation**: Catch regressions in operation performance

---

## Test Infrastructure

### Dependencies

```toml
[dev-dependencies]
wasm-bindgen-test = "0.3"
```

### Test Structure

```
tests/
├── wasm_test_utils/
│   └── mod.rs                       # Shared utilities for WASM tests
├── WASM_TEST_TEMPLATE.rs            # Template for new tests
├── wasm_threshold_tests.rs          # ✅ 17 tests (complete)
├── wasm_gaussian_blur_tests.rs      # ✅ 24 tests (complete)
└── wasm_[operation]_tests.rs        # 139 operations remaining
```

### Test Utilities (`tests/wasm_test_utils/mod.rs`)

**Test Image Creation:**
- `create_test_image_gray()` - 10x10 grayscale with gradient pattern
- `create_test_image_rgb()` - 10x10 RGB with gradient pattern
- `create_test_image_large()` - 100x100 checkerboard for performance tests

**Image Comparison:**
- `images_are_similar(img1, img2, tolerance)` - Compare with pixel-wise tolerance
- `check_dimensions(img, w, h, c)` - Verify shape
- `is_black(img)` - All pixels zero
- `is_white(img)` - All pixels 255

**Image Analysis:**
- `average_pixel_value(img)` - Mean pixel intensity
- `pixel_stddev(img)` - Standard deviation (smoothness measure)
- `count_nonzero(img)` - Non-zero pixel count

---

## Test Categories

Every WASM operation should have tests in these 8 categories:

### 1. Smoke Tests
**Purpose**: Verify function doesn't panic or crash

```rust
#[wasm_bindgen_test]
async fn test_[operation]_basic_smoke() {
    let src = create_test_image_rgb();
    let result = [operation_wasm](&src, params).await;
    assert!(result.is_ok(), "Should not panic");
}
```

**Requirements**:
- ✅ Function completes without panic
- ✅ Returns Ok or Err (not panic)

### 2. Dimension Tests
**Purpose**: Verify output shape matches expected dimensions

```rust
#[wasm_bindgen_test]
async fn test_[operation]_output_dimensions() {
    let src = create_test_image_rgb();
    let result = [operation_wasm](&src, params).await.unwrap();
    assert_eq!(result.width(), expected_width);
    assert_eq!(result.height(), expected_height);
    assert_eq!(result.channels(), expected_channels);
}
```

**Requirements**:
- ✅ Width correct
- ✅ Height correct
- ✅ Channels correct (grayscale→1, RGB→3)

### 3. Correctness Tests
**Purpose**: Verify operation produces mathematically correct output

**Examples by Operation Type:**

**Filters** (blur, gaussian, median):
```rust
// Should reduce standard deviation (smoother)
assert!(pixel_stddev(&blurred) < pixel_stddev(&src));

// Should preserve average intensity
let avg_diff = (average_pixel_value(&src) - average_pixel_value(&result)).abs();
assert!(avg_diff < 5.0);
```

**Thresholding** (threshold, adaptive_threshold):
```rust
// Should produce binary output
for &pixel in result.get_data().iter() {
    assert!(pixel == 0 || pixel == max_val);
}
```

**Edge Detection** (canny, sobel, scharr):
```rust
// Should detect edges (non-zero pixels)
assert!(count_nonzero(&result) > 0);

// Edges are sparser than original
assert!(count_nonzero(&result) < count_nonzero(&src));
```

**Morphology** (erode, dilate):
```rust
// Erode should reduce white areas
assert!(count_nonzero(&eroded) <= count_nonzero(&src));

// Dilate should increase white areas
assert!(count_nonzero(&dilated) >= count_nonzero(&src));
```

**Transforms** (resize, rotate, warp):
```rust
// Resized dimensions should match target
assert_eq!(result.width(), target_width);
assert_eq!(result.height(), target_height);

// Should preserve approximate average
let avg_diff = (average_pixel_value(&src) - average_pixel_value(&result)).abs();
assert!(avg_diff < 10.0);
```

### 4. Edge Cases
**Purpose**: Test boundary conditions and unusual inputs

```rust
#[wasm_bindgen_test]
async fn test_[operation]_small_image() {
    let data = vec![128u8; 3 * 3 * 1]; // 3x3 grayscale
    let src = WasmMat::from_image_data(&data, 3, 3, 1).unwrap();
    let result = [operation_wasm](&src, params).await;
    assert!(result.is_ok());
}

#[wasm_bindgen_test]
async fn test_[operation]_large_image() {
    let src = create_test_image_large(); // 100x100
    let result = [operation_wasm](&src, params).await;
    assert!(result.is_ok());
}

#[wasm_bindgen_test]
async fn test_[operation]_uniform_image() {
    let data = vec![128u8; 10 * 10 * 3]; // Uniform gray
    let src = WasmMat::from_image_data(&data, 10, 10, 3).unwrap();
    let result = [operation_wasm](&src, params).await.unwrap();
    // Verify operation handles uniform input gracefully
}
```

### 5. Parameter Tests
**Purpose**: Verify different parameter combinations work correctly

```rust
#[wasm_bindgen_test]
async fn test_[operation]_parameter_variations() {
    let src = create_test_image_rgb();

    // Test boundary values
    let result_min = [operation_wasm](&src, min_param).await;
    assert!(result_min.is_ok());

    let result_max = [operation_wasm](&src, max_param).await;
    assert!(result_max.is_ok());

    // Verify parameters have expected effect
    let result1 = [operation_wasm](&src, param1).await.unwrap();
    let result2 = [operation_wasm](&src, param2).await.unwrap();
    // Example: larger blur kernel should reduce stddev more
    assert!(pixel_stddev(&result2) < pixel_stddev(&result1));
}
```

### 6. Backend Tests
**Purpose**: Verify backend selection system works correctly

```rust
#[wasm_bindgen_test]
async fn test_[operation]_cpu_backend() {
    set_backend_wasm("cpu");
    let src = create_test_image_rgb();
    let result = [operation_wasm](&src, params).await;
    assert!(result.is_ok(), "Should work with CPU backend");
    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_[operation]_gpu_backend() {
    set_backend_wasm("webgpu");
    let src = create_test_image_rgb();
    let result = [operation_wasm](&src, params).await;
    // May fail if GPU unavailable - that's OK
    if let Ok(output) = result {
        assert_eq!(output.width(), src.width());
    }
    set_backend_wasm("auto");
}

#[wasm_bindgen_test]
async fn test_[operation]_cpu_gpu_consistency() {
    let src = create_test_image_rgb();

    set_backend_wasm("cpu");
    let cpu_result = [operation_wasm](&src, params).await.unwrap();

    set_backend_wasm("auto");
    let auto_result = [operation_wasm](&src, params).await.unwrap();

    // Adjust tolerance based on operation:
    // - Deterministic (threshold, flip): tolerance = 0.0
    // - Floating point (blur, filters): tolerance = 1.0-5.0
    assert!(images_are_similar(&cpu_result, &auto_result, 5.0));

    set_backend_wasm("auto");
}
```

### 7. OpenCV.js Parity
**Purpose**: Verify behavior matches OpenCV.js exactly

**Reference**: https://docs.opencv.org/4.x/d5/d0f/tutorial_js_table_of_contents_imgproc.html

```rust
#[wasm_bindgen_test]
async fn test_[operation]_opencv_js_parity() {
    // Document OpenCV.js behavior:
    // 1. What parameters does OpenCV.js accept?
    // 2. What are the default values?
    // 3. Are there any quirks or edge cases?
    // 4. What error cases does OpenCV.js handle?

    let src = create_test_image_rgb();
    let result = [operation_wasm](&src, params).await.unwrap();

    // Example checks:
    // - Verify parameter semantics match
    // - Verify return type matches
    // - Verify error handling matches
    assert!(check_dimensions(&result, expected_w, expected_h, expected_c));
}
```

### 8. Custom Tests
**Purpose**: Operation-specific validation

Add tests specific to the operation's unique behavior:

```rust
// For blur operations: test convergence
#[wasm_bindgen_test]
async fn test_gaussian_blur_idempotency_limit() {
    let mut current = create_test_image_rgb();
    for _ in 0..10 {
        current = gaussian_blur_wasm(&current, 7, 2.0).await.unwrap();
    }
    // After many blurs, should converge to uniform
    assert!(pixel_stddev(&current) < 30.0);
}

// For threshold: test exact boundary behavior
#[wasm_bindgen_test]
async fn test_threshold_exact_boundary() {
    let data = vec![128u8; 10 * 10];
    let src = WasmMat::from_image_data(&data, 10, 10, 1).unwrap();
    let result = threshold_wasm(&src, 128.0, 255.0).await.unwrap();
    // OpenCV behavior: pixel > thresh → max_val, else → 0
    assert!(is_black(&result), "Pixels at exact threshold should be 0");
}
```

---

## Running Tests

### Native Target (Fast - Unit Tests Only)
```bash
# Run all backend unit tests
cargo test --lib backend_native_test --features gpu

# Output: 17-18 tests pass in <1 second
```

### WASM Target (Full WASM Tests)
```bash
# Install wasm-pack (if not installed)
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Run WASM tests in browser
wasm-pack test --headless --chrome --features wasm

# Or Firefox
wasm-pack test --headless --firefox --features wasm

# Run specific test file
wasm-pack test --headless --chrome --features wasm -- --test wasm_threshold_tests
```

### Watch Mode (Development)
```bash
# Install cargo-watch
cargo install cargo-watch

# Watch and run native tests
cargo watch -x 'test --lib backend_native_test --features gpu'

# Watch and run WASM tests
cargo watch -s 'wasm-pack test --headless --chrome --features wasm'
```

---

## Current Test Coverage

| Operation | Tests | Status | Notes |
|-----------|-------|--------|-------|
| threshold_wasm | 17 | ✅ Complete | All categories covered |
| gaussian_blur_wasm | 24 | ✅ Complete | All categories + extras |
| **Remaining** | **139** | ❌ TODO | Use template |

**Progress**: 2/141 operations tested (1.4%)
**Target**: 141/141 operations (100%)

---

## Test Template Usage

### Step 1: Copy Template
```bash
cp tests/WASM_TEST_TEMPLATE.rs tests/wasm_resize_tests.rs
```

### Step 2: Find & Replace
```bash
# Replace placeholders:
[OPERATION_NAME] → resize
[operation_wasm] → resize_wasm
[PARAMS] → dst_width, dst_height
```

### Step 3: Customize Tests
- Add operation-specific correctness tests
- Adjust dimension assertions (resize changes dimensions!)
- Add parameter variation tests
- Document OpenCV.js behavior

### Step 4: Run Tests
```bash
wasm-pack test --headless --chrome --features wasm -- --test wasm_resize_tests
```

### Step 5: Update Coverage
Update this document's coverage table when complete.

---

## Example Test Progression

### Stage 1: Core Operations (Priority)
Test these 20 operations first (most commonly used):

1. ✅ threshold
2. ✅ gaussian_blur
3. ❌ resize
4. ❌ cvt_color_gray
5. ❌ canny
6. ❌ sobel
7. ❌ erode
8. ❌ dilate
9. ❌ flip
10. ❌ blur
11. ❌ median_blur
12. ❌ bilateral_filter
13. ❌ scharr
14. ❌ laplacian
15. ❌ adaptive_threshold
16. ❌ morphology_opening
17. ❌ morphology_closing
18. ❌ good_features_to_track
19. ❌ harris_corners
20. ❌ fast

### Stage 2: Drawing & Geometry (15 operations)
### Stage 3: Color Operations (11 operations)
### Stage 4: Morphology (11 operations)
### Stage 5: Arithmetic (10 operations)
### Stage 6: Feature Detection (10 operations)
### Stage 7: Machine Learning (8 operations)
### Stage 8: Remaining Operations (56 operations)

---

## Best Practices

### DO ✅
- **Write tests before refactoring** - catch regressions early
- **Test both CPU and GPU backends** - ensure consistency
- **Use descriptive test names** - `test_gaussian_blur_preserves_average` not `test_gb_avg`
- **Add comments explaining expected behavior** - future maintainers will thank you
- **Test edge cases** - small images, large images, uniform images
- **Verify OpenCV.js parity** - reference official OpenCV.js docs
- **Reset backend after each test** - always call `set_backend_wasm("auto")` in cleanup

### DON'T ❌
- **Don't skip backend tests** - they catch critical bugs
- **Don't use arbitrary tolerances** - understand why tolerance is needed
- **Don't test implementation details** - test observable behavior
- **Don't copy-paste tests without customization** - each operation is unique
- **Don't ignore failing tests** - fix or document why they fail
- **Don't assume GPU is available** - tests must pass without GPU

---

## Troubleshooting

### "Test failed: GPU not available"
This is OK for GPU backend tests. GPU tests should handle this gracefully:
```rust
#[wasm_bindgen_test]
async fn test_operation_gpu_backend() {
    set_backend_wasm("webgpu");
    let result = operation_wasm(&src, params).await;

    // OK if GPU unavailable
    if let Ok(output) = result {
        // Validate output
    }
    set_backend_wasm("auto");
}
```

### "Images not similar: diff = 10.5"
Adjust tolerance based on operation type:
- Deterministic operations: `tolerance = 0.0`
- Floating-point filters: `tolerance = 1.0-5.0`
- Complex operations: `tolerance = 5.0-10.0`

### "wasm-pack: command not found"
```bash
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

### "Test hangs indefinitely"
Check for:
- Infinite loops in operation
- Missing `.await` on async operation
- Deadlocks in backend selection

### "Dimensions don't match"
Some operations transform dimensions:
- `resize`: Changes width/height to target
- `threshold`: Converts to grayscale (channels=1)
- `rotate`: May change width/height depending on angle

Update assertions accordingly.

---

## Integration with Refactoring

When refactoring `src/wasm/mod.rs` (4731 lines → logical modules):

**Before moving any code:**
1. ✅ Verify test exists for that operation
2. ✅ Run test suite: `wasm-pack test --headless --chrome --features wasm`
3. ✅ Ensure all tests pass

**After moving code:**
1. ✅ Run test suite again
2. ✅ Verify no regressions
3. ✅ Update imports if needed

**If test fails after refactor:**
1. 🔴 **DO NOT COMMIT**
2. 🔴 Fix the regression
3. 🟢 Re-run tests until green
4. 🟢 Then commit

This workflow ensures zero regressions during the 6-8 week refactoring effort.

---

## Success Criteria

Before declaring "100% OpenCV.js API Parity":

- [ ] All 141 operations have test files
- [ ] Each operation has 8+ tests covering all categories
- [ ] All tests pass on WASM target
- [ ] CPU and GPU backends produce consistent results
- [ ] Documentation complete for each operation
- [ ] Coverage report shows 90%+ test coverage

**Current Progress**: 2/141 operations (1.4%)
**Estimated Effort**: 2-3 operations per day = 70 days = 14 weeks (3.5 months)

---

## References

- **OpenCV.js Docs**: https://docs.opencv.org/4.x/d5/d0f/tutorial_js_table_of_contents_imgproc.html
- **wasm-bindgen-test Guide**: https://rustwasm.github.io/wasm-bindgen/wasm-bindgen-test/index.html
- **Test Template**: `tests/WASM_TEST_TEMPLATE.rs`
- **Refactoring Analysis**: `docs/analysis/wasm-mod-refactoring.md`
- **Backend Design**: `docs/design/backend-gpu-cpu.md`

---

**Status**: Infrastructure complete, ready for test expansion
**Next Step**: Test Stage 1 operations (resize, cvt_color, canny, sobel, etc.)
