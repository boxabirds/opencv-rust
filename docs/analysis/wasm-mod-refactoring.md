# WASM Module Refactoring Analysis

**Date**: 2025-11-10
**File**: `src/wasm/mod.rs`
**Status**: 🔴 CRITICAL - Requires refactoring

---

## Current State: The Problem

### Size Metrics
- **4,731 lines** in a single file
- **141 async functions** exported to WASM
- **151 wasm_bindgen annotations**
- **~33.5 lines per function** average
- Becoming a serious liability for:
  - Code navigation
  - Compilation times
  - Merge conflicts
  - Maintenance
  - Testing

### Clippy Findings

**Critical Issues Found: 50+**

Common patterns of issues:
1. **Type mismatches** - i32 vs usize conversions (10+ instances)
2. **Missing arguments** - GPU function calls missing parameters (8+ instances)
3. **Borrow checker violations** - Mutable/immutable conflicts (5+ instances)
4. **Undefined types** - Missing imports or features (3+ instances)
5. **API signature mismatches** - CPU/GPU function signature differences

**Sample Critical Errors:**
```rust
// Line 4711: Type mismatch
let mut dst = Mat::new(rows, cols, num_channels, MatDepth::U8)
//                                  ^^^^^^^^^^^^ expected `usize`, found `i32`

// Line 4680: Missing argument
crate::gpu::ops::split_gpu_async(&src.inner).await
// Missing: &mut Vec<Mat> destination parameter

// Line 3552: Borrow conflict
let pixel = result.at_mut(row, col)  // mutable borrow
let gray = result.at(row, col)        // immutable borrow (same scope)

// Line 3711: Undefined type
let conv_layer = ConvLayer::new()  // ConvLayer not in scope
```

---

## Function Distribution by Category

Analysis of 141 functions shows clear logical groupings:

| Category | Count | Examples |
|----------|-------|----------|
| **Other/Misc** | 31 | akaze, anisotropic_diffusion, approx_poly_dp |
| **Morphology** | 11 | erode, dilate, opening, closing, gradient |
| **Color Conversion** | 11 | cvt_color_*, rgb_to_*, hsv_to_* |
| **Comparison/Bitwise** | 11 | bitwise_and/or/xor, compare, min/max |
| **Arithmetic** | 10 | add, subtract, multiply, divide, abs, pow |
| **Filtering** | 9 | blur, gaussian_blur, median_blur, bilateral |
| **Drawing** | 9 | draw_line, draw_rectangle, draw_circle |
| **Machine Learning** | 8 | svm, knn, decision_tree, neural_network |
| **Geometric** | 6 | resize, flip, rotate, warp_affine |
| **Feature Detection** | 6 | harris_corners, fast, orb, good_features |
| **Video Processing** | 5 | optical_flow, background_subtractor, tracker |
| **Edge Detection** | 4 | canny, sobel, scharr, laplacian |
| **Histogram** | 4 | calc_histogram, equalize, normalize |
| **Object Detection** | 4 | cascade, hog, aruco, qr_code |
| **Camera Calibration** | 3 | calibrate_camera, stereo_calibration |
| **Threshold** | 2 | threshold, adaptive_threshold |
| **Contour** | 2 | find_contours, contour_area |
| **Segmentation** | 2 | kmeans, watershed |
| **DNN** | 2 | load_network, blob_from_image |
| **Channel Ops** | 1 | merge_channels, split_channels |

---

## Recommended Refactoring Structure

### Phase 1: Create Module Structure (No Code Changes Yet)

```
src/wasm/
├── mod.rs              (200-300 lines - module declarations, WasmMat, init functions)
├── backend.rs          (172 lines - already extracted ✅)
│
├── core/              (15-20 functions)
│   ├── mod.rs
│   ├── filtering.rs    (9 functions: blur, gaussian_blur, median_blur, etc.)
│   ├── edge.rs         (4 functions: canny, sobel, scharr, laplacian)
│   └── threshold.rs    (2 functions: threshold, adaptive_threshold)
│
├── imgproc/           (60-70 functions)
│   ├── mod.rs
│   ├── morphology.rs   (11 functions: erode, dilate, opening, closing, etc.)
│   ├── color.rs        (11 functions: cvt_color_*, rgb_to_*, hsv_to_*)
│   ├── geometric.rs    (6 functions: resize, flip, rotate, warp_affine, etc.)
│   ├── drawing.rs      (9 functions: draw_line, draw_rectangle, etc.)
│   ├── histogram.rs    (4 functions: calc_histogram, equalize, etc.)
│   └── contour.rs      (2 functions: find_contours, contour_area)
│
├── features/          (10-15 functions)
│   ├── mod.rs
│   ├── detection.rs    (6 functions: harris_corners, fast, orb, etc.)
│   └── object.rs       (4 functions: cascade, hog, aruco, qr)
│
├── ml/                (8 functions)
│   ├── mod.rs
│   └── classifiers.rs  (svm, knn, decision_tree, neural_network, etc.)
│
├── video/             (5 functions)
│   ├── mod.rs
│   └── tracking.rs     (optical_flow, background_subtractor, trackers)
│
├── calib3d/           (3 functions)
│   ├── mod.rs
│   └── camera.rs       (calibrate_camera, stereo_calibration, etc.)
│
├── dnn/               (2 functions)
│   ├── mod.rs
│   └── network.rs      (load_network, blob_from_image)
│
├── segmentation/      (2 functions)
│   ├── mod.rs
│   └── cluster.rs      (kmeans, watershed)
│
├── arithmetic/        (10 functions)
│   ├── mod.rs
│   └── ops.rs          (add, subtract, multiply, divide, abs, pow, etc.)
│
├── comparison/        (11 functions)
│   ├── mod.rs
│   └── bitwise.rs      (bitwise_and/or/xor, compare, min/max, etc.)
│
└── misc/              (31 functions)
    ├── mod.rs
    └── various.rs      (akaze, anisotropic_diffusion, etc.)
```

### Benefits of This Structure

1. **Logical organization** - Mirrors OpenCV's module structure
2. **Smaller files** - 5-15 functions per file (~150-500 lines)
3. **Easy navigation** - Find functions by category
4. **Reduced merge conflicts** - Changes isolated to specific modules
5. **Faster compilation** - Parallel compilation of smaller modules
6. **Better testability** - Test modules independently
7. **100% OpenCV.js compatibility maintained** - All functions remain exported

---

## Safe Refactoring Strategy

### ⚠️ CRITICAL: Test Coverage First

**DO NOT REFACTOR WITHOUT TESTS**

Before moving ANY code, we need comprehensive unit tests:

```
Current Test Coverage: UNKNOWN (likely <10% for WASM functions)
Target: 90%+ coverage before refactoring
```

### Step 1: Add Unit Tests (Week 1-2)

**Goal**: Achieve 90%+ test coverage on critical functions

```rust
// Example test structure
#[cfg(test)]
#[cfg(target_arch = "wasm32")]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    async fn test_gaussian_blur_smoke() {
        // Test that function doesn't panic
        let src = WasmMat::new(100, 100, 3).unwrap();
        let result = gaussian_blur_wasm(&src, 5, 1.5).await;
        assert!(result.is_ok());
    }

    #[wasm_bindgen_test]
    async fn test_gaussian_blur_output_size() {
        // Test output dimensions match input
        let src = WasmMat::new(100, 100, 3).unwrap();
        let result = gaussian_blur_wasm(&src, 5, 1.5).await.unwrap();
        assert_eq!(result.width(), 100);
        assert_eq!(result.height(), 100);
    }

    #[wasm_bindgen_test]
    async fn test_gaussian_blur_backend_selection() {
        // Test backend selection works
        set_backend_wasm("cpu");
        let src = WasmMat::new(100, 100, 3).unwrap();
        let cpu_result = gaussian_blur_wasm(&src, 5, 1.5).await.unwrap();

        set_backend_wasm("auto");
        let auto_result = gaussian_blur_wasm(&src, 5, 1.5).await.unwrap();

        // Results should be similar (within tolerance)
        assert_images_similar(&cpu_result, &auto_result, 1.0);
    }
}
```

**Priority Functions to Test First:**
1. ✅ gaussian_blur_wasm (already has backend selection)
2. ✅ threshold_wasm (already has backend selection)
3. resize_wasm
4. cvt_color_gray_wasm
5. canny_wasm
6. sobel_wasm
7. erode_wasm
8. dilate_wasm
9. flip_wasm
10. blur_wasm

**Test Infrastructure Needed:**
```bash
# Add to Cargo.toml
[dev-dependencies]
wasm-bindgen-test = "0.3"

# Create test utilities
mkdir -p tests/wasm_utils
# tests/wasm_utils/mod.rs - helper functions for WASM testing
```

### Step 2: Document Current API (Week 1)

Before refactoring, document all 141 function signatures:

```bash
# Generate API documentation
cargo doc --no-deps --target wasm32-unknown-unknown --features wasm

# Create API compatibility checklist
src/wasm/mod.rs  → 141 functions ✓
src/wasm/core/filtering.rs  → 9 functions (after refactor)
src/wasm/imgproc/color.rs    → 11 functions (after refactor)
# etc.
```

### Step 3: Refactor Module by Module (Week 3-6)

**One module at a time, following this checklist:**

- [ ] Select module (start with smallest, e.g., threshold.rs - 2 functions)
- [ ] Verify 90%+ test coverage for those functions
- [ ] Create new file: `src/wasm/core/threshold.rs`
- [ ] Copy functions from mod.rs → threshold.rs
- [ ] Add module declaration in mod.rs
- [ ] Run tests: `cargo test --target wasm32-unknown-unknown --features wasm`
- [ ] Run clippy: `cargo clippy --target wasm32-unknown-unknown --features wasm`
- [ ] Verify WASM build: `wasm-pack build --target web --features wasm`
- [ ] Run gallery smoke tests (manual)
- [ ] Commit changes
- [ ] Move to next module

**Order of Refactoring (Smallest → Largest):**
1. threshold.rs (2 functions) - ✅ Already has backend selection
2. dnn/network.rs (2 functions)
3. segmentation/cluster.rs (2 functions)
4. contour.rs (2 functions)
5. calib3d/camera.rs (3 functions)
6. edge.rs (4 functions)
7. histogram.rs (4 functions)
8. video/tracking.rs (5 functions)
9. geometric.rs (6 functions)
10. features/detection.rs (6 functions)
... continue in size order

---

## Risks & Mitigation

### Risk 1: Breaking WASM Bindings

**Risk**: Function moves might break wasm-bindgen exports

**Mitigation**:
- Keep all functions as `pub` exports
- Use `#[wasm_bindgen(js_name = "functionName")]` consistently
- Verify wasm-pack build after each module refactor
- Test in gallery after each major change

### Risk 2: Merge Conflicts During Long Refactor

**Risk**: Refactoring takes 4-6 weeks, conflicts with other work

**Mitigation**:
- Create dedicated refactoring branch: `refactor/wasm-modules`
- Refactor in small PRs (1-2 modules per PR)
- Rebase frequently on main branch
- Freeze wasm/mod.rs changes during refactor period

### Risk 3: Test Coverage is Low

**Risk**: Moving code without tests might introduce silent bugs

**Mitigation**:
- ⚠️ **DO NOT refactor any function without tests**
- Start with OpenCV.js parity tests (Priority 2 from plan)
- Use gallery as integration test suite
- Add smoke tests for all 141 functions before refactoring

### Risk 4: Clippy Issues Multiply During Refactor

**Risk**: Moving broken code spreads problems across files

**Mitigation**:
- Fix clippy issues BEFORE moving code
- Run `cargo clippy --fix` on mod.rs first
- Set up CI to block PRs with clippy warnings
- Fix type mismatches and missing arguments first

---

## Recommended Action Plan

### Option A: Test-First Refactoring (RECOMMENDED) ✅

**Timeline**: 6-8 weeks
**Risk**: Low
**Benefit**: High confidence, no regressions

1. **Week 1-2**: Add comprehensive unit tests to existing mod.rs
   - Target: 90%+ coverage on critical 20 functions
   - Use wasm-bindgen-test framework
   - Create test utilities and helpers

2. **Week 3**: Fix all clippy issues in mod.rs
   - Run `cargo clippy --fix --allow-dirty`
   - Manually fix type mismatches
   - Verify all 141 functions compile

3. **Week 4-6**: Refactor modules (smallest → largest)
   - 2-3 modules per week
   - Small PRs (1 module = 1 PR)
   - Run tests + clippy after each move

4. **Week 7-8**: Verify & document
   - Gallery integration testing
   - Update documentation
   - Create migration guide for future contributors

### Option B: Fix-In-Place (NOT RECOMMENDED) ⚠️

**Timeline**: 2-3 weeks
**Risk**: High
**Benefit**: Faster but dangerous

1. Fix clippy issues in current mod.rs
2. No structural changes
3. Add tests gradually

**Why not recommended:**
- Doesn't solve the 4731-line file problem
- Compilation times remain slow
- Merge conflicts continue
- Technical debt accumulates

### Option C: Hybrid Approach (COMPROMISE)

**Timeline**: 4-5 weeks
**Risk**: Medium
**Benefit**: Balanced

1. **Week 1**: Fix critical clippy issues (type mismatches, missing args)
2. **Week 2**: Add tests for core operations (filtering, edge, threshold)
3. **Week 3-4**: Refactor only core modules (filtering, edge, threshold, color)
4. **Week 5**: Verify core refactor, defer rest to later

**Result**: Reduces file to ~3000 lines, fixes most critical issues

---

## Recommendation

**Choose Option A: Test-First Refactoring**

**Rationale:**
1. The file is too large and complex to refactor safely without tests
2. Tests are needed anyway (Priority 2: OpenCV.js API Parity)
3. Proper refactoring now saves months of maintenance pain later
4. Backend selection is already implemented - perfect time to refactor
5. 6-8 weeks is acceptable for 4731 lines of critical WASM code

**First Steps (This Week):**
1. Set up wasm-bindgen-test infrastructure
2. Write tests for 2 functions already migrated to backend selection:
   - ✅ gaussian_blur_wasm
   - ✅ threshold_wasm
3. Create test template for remaining 139 functions
4. Run clippy and document all issues in tracking sheet

---

## Success Criteria

After refactoring is complete:

✅ **Code Organization**
- No file exceeds 500 lines
- Logical module structure mirrors OpenCV
- Easy to find any function

✅ **Quality**
- 90%+ test coverage on WASM functions
- Zero clippy warnings
- All 141 functions have unit tests

✅ **Compatibility**
- 100% OpenCV.js API parity maintained
- All gallery demos work unchanged
- WASM build size unchanged or smaller

✅ **Maintainability**
- Clear contribution guidelines
- Each module is independently testable
- New functions have obvious home

---

## Current Status

**Completed:**
- ✅ Backend selection system (src/wasm/backend.rs)
- ✅ Design document (docs/design/backend-gpu-cpu.md)
- ✅ 2 functions migrated to backend selection (gaussian_blur, threshold)
- ✅ Clippy analysis completed (50+ issues identified)
- ✅ Function categorization completed (141 functions grouped)

**Next Actions:**
1. 🔴 **CRITICAL**: Set up test infrastructure (wasm-bindgen-test)
2. 🔴 **CRITICAL**: Write tests for gaussian_blur_wasm and threshold_wasm
3. 🟡 Create test template and expand to 20 core functions
4. 🟡 Fix clippy type mismatches and missing arguments
5. 🟢 Begin refactoring core module (threshold.rs → 2 functions)

---

**Status**: Refactoring analysis complete, awaiting decision on approach
**Recommendation**: Option A (Test-First Refactoring)
**Timeline**: 6-8 weeks
**Next Step**: Set up wasm-bindgen-test and write first 2 tests
