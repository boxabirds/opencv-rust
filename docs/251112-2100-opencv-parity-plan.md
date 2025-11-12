# OpenCV.js Parity Testing Implementation Plan

**Date:** 2025-11-12
**Status:** Gap Analysis & Implementation Roadmap

## Executive Summary

This project claims OpenCV.js API parity but lacks **automated bit-level comparison tests** against opencv.js. The infrastructure exists but is **90% incomplete**. This document catalogues the gap and provides an implementation plan for headless WebGPU testing.

---

## Current State: Gap Analysis

### What EXISTS

#### 1. WASM Function Coverage
- **141 functions exposed** via wasm-bindgen
- All major OpenCV operations implemented (filters, transforms, feature detection, ML, etc.)
- GPU acceleration via WebGPU for core operations

#### 2. Native CPU Accuracy Tests
Location: `tests/test_accuracy_*.rs` (20 files)

These test determinism and correctness of **CPU implementations** only:
- gaussian_blur, threshold, sobel, resize, flip, etc.
- Use `test_utils::assert_images_equal()` for bit-exact comparison
- Run natively (not WASM, not against opencv.js)

#### 3. WASM Self-Comparison Tests
Location: `tests/wasm_*_tests.rs` (7 files)

These test **GPU vs CPU within this project**:
- gaussian_blur, threshold, sobel, resize, flip, erode, dilate
- Verify GPU produces similar results to CPU (tolerance: 5.0)
- Run in-browser via wasm-bindgen-test
- **Do NOT compare against opencv.js**

#### 4. OpenCV.js Comparison Skeleton (INCOMPLETE)
Location: `tests/opencv_js_reference/`

```
generate_tests.js    (439 lines) - Config defined, implementation STUBBED
compare_apis.js      (255 lines) - API comparison utilities
benchmark_suite.js   (370 lines) - Performance benchmarking
```

**Test Configs Defined** (12 operations):
1. `gaussian_blur` - tolerance: max_pixel_diff=1, max_mean_diff=0.1
2. `resize` - tolerance: 2px, 0.5 mean
3. `threshold` - tolerance: 0 (exact match)
4. `canny` - tolerance: 1px, 5% outliers
5. `sobel` - tolerance: 2px, 3% outliers
6. `erode` - tolerance: 0px, 0.5% outliers
7. `dilate` - tolerance: 0px, 0.5% outliers
8. `bilateral_filter` - tolerance: 3px, 5% outliers
9. `median_blur` - tolerance: 0px, 0.5% outliers
10. `laplacian` - tolerance: 2px, 3% outliers
11. `flip` - tolerance: 0 (exact match)
12. `adaptive_threshold` - tolerance: 1px, 2% outliers

**Critical Code in generate_tests.js:286-291** (STUBBED):
```javascript
// In a real implementation, we would:
// 1. Load both images
// 2. Compare pixel by pixel
// 3. Calculate differences (max, mean, outliers)
// 4. Apply tolerance thresholds

console.log(`  Comparing ${refFile}`);
comparison.passed++;  // FAKE - always passes!
```

---

### What's MISSING

#### 1. Actual Pixel Comparison Implementation
The comparison code is **completely stubbed out**. No images are loaded, no pixels are compared, tests always pass.

#### 2. OpenCV.js Integration
- No code to load opencv.js in Node.js/headless browser
- No code to run opencv.js operations
- No reference output generation

#### 3. Headless Browser Test Infrastructure
- No Playwright/Puppeteer setup
- No CI integration
- No headless WebGPU configuration

#### 4. Coverage Gap
- **12/141 operations have test configs defined** (8.5%)
- **0/141 operations have working comparison tests** (0%)
- **129 operations completely uncovered**

---

## Implementation Plan

### Phase 1: Prove the Concept (1-2 days)

**Goal:** Get ONE operation (gaussian_blur) working end-to-end with headless WebGPU.

#### 1.1 Set Up Headless Browser Environment
```bash
npm install --save-dev playwright @playwright/test
```

**Why Playwright over Puppeteer:**
- Better WebGPU support in Chromium
- Built-in test runner
- Better async handling

#### 1.2 Enable Headless WebGPU
```javascript
// tests/opencv_js_reference/playwright.config.js
export default {
  use: {
    launchOptions: {
      args: [
        '--enable-unsafe-webgpu',           // Enable WebGPU
        '--enable-features=Vulkan',         // GPU backend
        '--use-angle=swiftshader',          // Software GPU (reliable)
        '--headless',
      ],
    },
  },
};
```

**Note:** SwiftShader provides software-based WebGPU that works reliably in headless mode.

#### 1.3 Implement Pixel Comparison
```javascript
// tests/opencv_js_reference/pixel_comparison.js
export function compareImages(imageA, imageB, tolerance) {
  const dataA = imageA.data;
  const dataB = imageB.data;

  let maxDiff = 0;
  let totalDiff = 0;
  let outlierCount = 0;
  let diffCount = 0;

  for (let i = 0; i < dataA.length; i++) {
    const diff = Math.abs(dataA[i] - dataB[i]);
    if (diff > 0) {
      diffCount++;
      totalDiff += diff;
      maxDiff = Math.max(maxDiff, diff);

      if (diff > tolerance.max_pixel_diff) {
        outlierCount++;
      }
    }
  }

  const meanDiff = diffCount > 0 ? totalDiff / diffCount : 0;
  const outlierPercent = (outlierCount / dataA.length) * 100;

  return {
    maxDiff,
    meanDiff,
    outlierPercent,
    passed: (
      maxDiff <= tolerance.max_pixel_diff * 2 &&  // Allow 2x max for outliers
      meanDiff <= tolerance.max_mean_diff &&
      outlierPercent <= tolerance.max_outliers_percent
    ),
  };
}
```

#### 1.4 Create Single-Operation Test
```javascript
// tests/opencv_js_reference/test_gaussian_blur.spec.js
import { test, expect } from '@playwright/test';
import { compareImages } from './pixel_comparison.js';

test('gaussian_blur matches opencv.js', async ({ page }) => {
  // Load test page with both libraries
  await page.goto('http://localhost:8080/test-harness.html');

  // Run both implementations
  const result = await page.evaluate(async () => {
    // Load test image
    const img = await loadTestImage('lenna.png');

    // Run opencv.js
    const cvMat = cv.matFromImageData(img);
    const cvDst = new cv.Mat();
    cv.GaussianBlur(cvMat, cvDst, new cv.Size(5, 5), 1.5);
    const cvResult = getImageDataFromMat(cvDst);

    // Run our WASM
    const wasmMat = await WasmMat.fromImageData(img.data, img.width, img.height, 4);
    const wasmDst = await gaussianBlur(wasmMat, 5, 1.5);
    const wasmResult = wasmDst.getData();

    return {
      opencv: Array.from(cvResult.data),
      ours: Array.from(wasmResult),
      width: img.width,
      height: img.height,
    };
  });

  // Compare pixel-by-pixel
  const comparison = compareImages(
    { data: result.opencv },
    { data: result.ours },
    { max_pixel_diff: 1, max_mean_diff: 0.1, max_outliers_percent: 1.0 }
  );

  console.log('Comparison:', comparison);
  expect(comparison.passed).toBe(true);
});
```

#### 1.5 Test Infrastructure Files
```
tests/opencv_js_reference/
├── playwright.config.js       (NEW - Playwright config)
├── pixel_comparison.js        (NEW - Comparison logic)
├── test_gaussian_blur.spec.js (NEW - First real test)
├── test-harness.html          (NEW - Loads both libraries)
├── package.json               (UPDATE - Add Playwright)
└── README.md                  (UPDATE - Usage instructions)
```

#### 1.6 Success Criteria
- ✅ Test runs headless with WebGPU
- ✅ Loads both opencv.js and our WASM
- ✅ Compares gaussian_blur pixel-by-pixel
- ✅ Reports actual differences (max, mean, outliers)
- ✅ Passes or fails based on tolerance

---

### Phase 2: Expand Core Operations (3-5 days)

#### 2.1 Implement Tests for Configured Operations
Complete the 12 operations that already have tolerance configs:
1. gaussian_blur ✅ (from Phase 1)
2. resize
3. threshold
4. canny
5. sobel
6. erode
7. dilate
8. bilateral_filter
9. median_blur
10. laplacian
11. flip
12. adaptive_threshold

**Template:**
```javascript
// tests/opencv_js_reference/test_{operation}.spec.js
import { test, expect } from '@playwright/test';
import { compareImages } from './pixel_comparison.js';
import { TEST_CONFIGS } from './generate_tests.js';

const config = TEST_CONFIGS.{operation};

for (const params of config.params) {
  test(`{operation} with params ${JSON.stringify(params)}`, async ({ page }) => {
    // Run comparison
    const comparison = await runComparison(page, '{operation}', params);
    expect(comparison.passed).toBe(true);
  });
}
```

#### 2.2 Test Fixtures
```
tests/fixtures/
├── lenna.png        (512x512 classic test image)
├── shapes.png       (geometric shapes)
├── text.png         (text rendering)
├── gradient.png     (smooth gradients)
├── noise.png        (random noise)
└── edges.png        (sharp edges)
```

#### 2.3 CI Integration
```yaml
# .github/workflows/opencv-parity-tests.yml
name: OpenCV.js Parity Tests

on: [push, pull_request]

jobs:
  parity-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'

      - name: Install dependencies
        run: |
          cd tests/opencv_js_reference
          npm install

      - name: Build WASM
        run: wasm-pack build --target web --out-dir pkg --features gpu,wasm

      - name: Install Playwright browsers
        run: npx playwright install --with-deps chromium

      - name: Run parity tests
        run: |
          cd tests/opencv_js_reference
          npx playwright test

      - name: Upload test report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: parity-test-report
          path: tests/opencv_js_reference/playwright-report/
```

---

### Phase 3: Comprehensive Coverage (1-2 weeks)

#### 3.1 Categorize Remaining Operations

**High Priority** (Core operations used everywhere):
- Color conversions: cvtColorGray, cvtColorHSV, rgbToGray, etc. (8 ops)
- Basic filters: boxBlur, medianBlur, bilateralFilter (already covered)
- Morphology: erode, dilate, opening, closing (partially covered)
- Geometric: rotate, warpAffine, warpPerspective (3 ops)
- Arithmetic: add, subtract, multiply, divide, absdiff (5 ops)

**Medium Priority** (Feature detection, advanced):
- Edge detection: scharr (sobel/canny covered)
- Features: harris, goodFeaturesToTrack, FAST, SIFT, ORB (5 ops)
- Contours: findContours, boundingRect, contourArea (3 ops)
- Transforms: integral, histogram (2 ops)

**Low Priority** (Specialized/ML):
- Tracking: meanshift, camshift, KCF, CSRT (4 ops)
- ML: SVM, decision trees, neural networks (3 ops)
- Calibration: calibrateCamera, stereoCalibration (2 ops)
- Advanced: Hough transforms, watershed, inpainting (5 ops)

**Total:** ~50 high-priority operations to cover

#### 3.2 Tolerance Configuration Strategy

Some operations need **different tolerances** based on algorithm characteristics:

**Exact Match Required (tolerance = 0):**
- Threshold (binary decision)
- Flip (geometric reordering)
- Logical operations (bitwise AND/OR/XOR)

**Low Tolerance (1-2px):**
- Gaussian blur (floating-point kernel)
- Resize (interpolation)
- Color conversions (rounding)

**Medium Tolerance (3-5px):**
- Edge detection (gradient computation)
- Morphological operations (neighborhood)
- Advanced filters (bilateral, NLM)

**High Tolerance (5-10px, 5-10% outliers):**
- Feature detection (SIFT, ORB - implementation-dependent)
- ML operations (different training/optimization)

#### 3.3 Automated Tolerance Discovery

For operations without defined tolerances, use **empirical measurement**:

```javascript
// tests/opencv_js_reference/discover_tolerances.js
async function discoverTolerance(operation, testImages) {
  const results = [];

  for (const image of testImages) {
    const comparison = await runComparison(operation, image);
    results.push({
      image: image.name,
      maxDiff: comparison.maxDiff,
      meanDiff: comparison.meanDiff,
      outliers: comparison.outlierPercent,
    });
  }

  // Calculate 95th percentile tolerances
  const tolerances = {
    max_pixel_diff: percentile(results.map(r => r.maxDiff), 95),
    max_mean_diff: percentile(results.map(r => r.meanDiff), 95),
    max_outliers_percent: percentile(results.map(r => r.outliers), 95),
  };

  console.log(`Suggested tolerances for ${operation}:`, tolerances);
  return tolerances;
}
```

Run this on a diverse image set to establish baseline tolerances for each operation.

---

### Phase 4: Maintenance & Monitoring (Ongoing)

#### 4.1 Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

# Run parity tests for changed operations
CHANGED_OPS=$(git diff --cached --name-only | grep "src/gpu/ops" | sed 's/.*\///' | sed 's/\.rs//')

if [ -n "$CHANGED_OPS" ]; then
  echo "Running parity tests for changed operations..."
  for op in $CHANGED_OPS; do
    npm test -- test_${op}.spec.js
  done
fi
```

#### 4.2 Regression Detection
Track differences over time:
```javascript
// Store baseline comparison results
const baseline = {
  operation: 'gaussian_blur',
  commit: 'abc123',
  maxDiff: 0.8,
  meanDiff: 0.05,
  outliers: 0.2,
};

// On subsequent runs, alert if differences increase
if (current.maxDiff > baseline.maxDiff * 1.5) {
  console.warn('⚠️  Regression detected: maxDiff increased significantly');
}
```

#### 4.3 Performance Tracking
```javascript
// Benchmark both implementations
const opencvTime = await measureTime(() => cv.GaussianBlur(...));
const oursTime = await measureTime(() => gaussianBlur(...));
const speedup = opencvTime / oursTime;

console.log(`Speedup: ${speedup.toFixed(2)}x`);

// Fail if we're slower than opencv.js CPU
if (speedup < 1.0) {
  console.error('❌ Performance regression: slower than opencv.js');
}
```

---

## Technical Considerations

### Headless WebGPU Support

**Chrome/Chromium:**
```bash
chromium \
  --headless \
  --enable-unsafe-webgpu \
  --use-angle=swiftshader \
  --disable-vulkan-fallback-to-gl-for-testing
```

**Firefox:**
Limited WebGPU support in headless mode. Chrome recommended.

**SwiftShader vs Hardware GPU:**
- **SwiftShader**: Software renderer, consistent results, slower
- **Hardware GPU**: Fast, but varies by machine (driver differences)
- **Recommendation**: Use SwiftShader for CI, hardware for local dev

### Known Differences OpenCV.js vs OpenCV-Rust

#### 1. Border Handling
OpenCV.js uses `BORDER_DEFAULT` (reflect-101). Our implementation may differ. **Action:** Document and test border cases explicitly.

#### 2. Floating-Point Precision
JavaScript uses 64-bit floats, WASM uses 32-bit. **Action:** Tolerances account for this (max_pixel_diff ≥ 1).

#### 3. SIMD/GPU Optimizations
Different implementations may produce slightly different results due to:
- Multiply-accumulate ordering
- Intermediate rounding
- Parallel reduction algorithms

**Action:** Tolerances set empirically to allow implementation differences while catching bugs.

#### 4. Color Space Assumptions
OpenCV.js assumes BGR, we may use RGB. **Action:** Verify conversions are correct.

---

## Deliverables & Timeline

### Week 1: Foundation
- ✅ Phase 1 complete: gaussian_blur working end-to-end
- ✅ CI workflow passing
- ✅ Documentation updated

### Week 2: Core Coverage
- ✅ 12 configured operations tested
- ✅ Test fixtures created
- ✅ Tolerance discovery tool working

### Week 3: Expansion
- ✅ 50 high-priority operations covered
- ✅ Automated tolerance discovery for remaining ops
- ✅ Regression detection system

### Week 4: Polish
- ✅ Pre-commit hooks
- ✅ Performance tracking
- ✅ Comprehensive documentation

---

## Success Metrics

1. **Coverage:** 90% of WASM operations have comparison tests
2. **Pass Rate:** 95% of tests pass within defined tolerances
3. **CI Time:** Parity tests complete in < 10 minutes
4. **Regression Detection:** Any pixel-level regression caught within 1 commit
5. **Documentation:** Every operation has documented tolerance rationale

---

## Risks & Mitigations

### Risk 1: Headless WebGPU Flakiness
**Mitigation:** Use SwiftShader, retry failed tests 3x, collect screenshots on failure.

### Risk 2: Tolerance Tuning
**Mitigation:** Start conservative (higher tolerances), tighten over time as implementations improve.

### Risk 3: CI Resource Usage
**Mitigation:** Run full suite nightly, subset on PRs (only changed operations).

### Risk 4: OpenCV.js Version Drift
**Mitigation:** Pin opencv.js version, document which version we target parity with.

---

## Open Questions

1. **Which OpenCV.js version?** 4.8.0 (latest stable) or 4.5.5 (more widely deployed)?
2. **Hardware GPU in CI?** GitHub Actions has GPU runners, but expensive. Start with SwiftShader.
3. **Test image licensing?** Ensure test fixtures are properly licensed for open-source use.

---

## Appendix A: Function Coverage Matrix

### Operations WITH Test Configs (12)
| Operation | GPU | CPU | Test Config | Tolerance | Status |
|-----------|-----|-----|-------------|-----------|--------|
| gaussian_blur | ✅ | ✅ | ✅ | 1px/0.1mean | ⚠️ Stubbed |
| resize | ✅ | ✅ | ✅ | 2px/0.5mean | ⚠️ Stubbed |
| threshold | ✅ | ✅ | ✅ | 0px (exact) | ⚠️ Stubbed |
| canny | ✅ | ✅ | ✅ | 1px/5%out | ⚠️ Stubbed |
| sobel | ✅ | ✅ | ✅ | 2px/3%out | ⚠️ Stubbed |
| erode | ✅ | ✅ | ✅ | 0px/0.5%out | ⚠️ Stubbed |
| dilate | ✅ | ✅ | ✅ | 0px/0.5%out | ⚠️ Stubbed |
| bilateral_filter | ✅ | ✅ | ✅ | 3px/5%out | ⚠️ Stubbed |
| median_blur | ✅ | ✅ | ✅ | 0px/0.5%out | ⚠️ Stubbed |
| laplacian | ✅ | ✅ | ✅ | 2px/3%out | ⚠️ Stubbed |
| flip | ✅ | ✅ | ✅ | 0px (exact) | ⚠️ Stubbed |
| adaptive_threshold | ✅ | ✅ | ✅ | 1px/2%out | ⚠️ Stubbed |

### Operations WITHOUT Test Configs (129)
Sampling of high-priority uncovered operations:

**Filters (12):**
- box_blur, blur, scharr, filter2d, guided_filter, gabor_filter
- nlm_denoising, anisotropic_diffusion, distance_transform
- pyr_up, pyr_down, integral_image

**Color Conversions (8):**
- cvt_color_gray, cvt_color_hsv, cvt_color_lab, cvt_color_ycrcb
- rgb_to_*, hsv_to_rgb, lab_to_rgb, ycrcb_to_rgb

**Geometric (7):**
- rotate, warp_affine, warp_perspective, remap
- get_rotation_matrix_2d

**Arithmetic (10):**
- add, subtract, multiply, divide, absdiff
- add_weighted, convert_scale, normalize
- exp, log, sqrt, pow, min, max

**Morphology (6):**
- morphology_opening, morphology_closing
- morphology_gradient, morphology_tophat, morphology_blackhat

**Feature Detection (10):**
- harris_corners, good_features_to_track, fast
- sift, orb, brisk, akaze, kaze

**Contours (5):**
- find_contours, bounding_rect, contour_area
- approx_poly_dp, arc_length, convex_hull

**Comparison/Logic (6):**
- bitwise_and, bitwise_or, bitwise_xor, bitwise_not
- in_range, lut

**Histogram (4):**
- calc_histogram, equalize_histogram
- normalize_histogram, compare_histograms, back_projection

**Hough Transforms (3):**
- hough_lines, hough_lines_p, hough_circles

**Advanced (20+):**
- watershed, inpaint, kmeans, moments
- detect_aruco, detect_qr
- Tracking: meanshift, camshift, mosse, csrt, kcf
- ML: svm, decision_tree, random_forest, knn, neural_network
- Calibration: calibrate_camera, fisheye, stereo, solve_pnp
- Stitching: panorama_stitcher, feather_blender, multiband_blender
- DNN: load_network, blob_from_image
- Optical flow: farneback

---

## Appendix B: Reference Implementation Pseudocode

```javascript
// Complete end-to-end test flow

async function testOperation(opName, params, testImage) {
  // 1. Load opencv.js (if not loaded)
  if (!window.cv) {
    await loadOpenCVJs();
  }

  // 2. Load our WASM (if not loaded)
  if (!window.opencvRust) {
    await init('./pkg/opencv_rust.js');
    await initGpu();
  }

  // 3. Load test image
  const imgData = await loadImage(testImage);

  // 4. Run opencv.js
  const cvMat = cv.matFromImageData(imgData);
  const cvDst = new cv.Mat();
  const cvStart = performance.now();
  runOpenCVJsOperation(opName, cvMat, cvDst, params);
  const cvTime = performance.now() - cvStart;
  const cvResult = getImageDataFromMat(cvDst);

  // 5. Run our WASM
  const wasmMat = WasmMat.fromImageData(imgData.data, imgData.width, imgData.height, 4);
  const wasmStart = performance.now();
  const wasmDst = await runOurOperation(opName, wasmMat, params);
  const wasmTime = performance.now() - wasmStart;
  const wasmResult = wasmDst.getData();

  // 6. Compare pixel-by-pixel
  const comparison = comparePixels(cvResult, wasmResult, TOLERANCES[opName]);

  // 7. Report results
  return {
    operation: opName,
    params,
    testImage,
    opencvTime: cvTime,
    ourTime: wasmTime,
    speedup: cvTime / wasmTime,
    comparison: {
      passed: comparison.passed,
      maxDiff: comparison.maxDiff,
      meanDiff: comparison.meanDiff,
      outlierPercent: comparison.outlierPercent,
    },
  };
}
```

---

## Conclusion

The project has **good bones** but is **critically incomplete** in the area that matters most: proving opencv.js parity. Implementing this plan will:

1. **Catch regressions** before they ship
2. **Build confidence** in API compatibility
3. **Document differences** between implementations
4. **Enable performance tracking** over time
5. **Support claims** of opencv.js parity with data

**Estimated effort:** 3-4 weeks for comprehensive coverage, 1 week for proof of concept.

**Recommendation:** Start with Phase 1 immediately. This is foundational infrastructure that should have existed from day one.
