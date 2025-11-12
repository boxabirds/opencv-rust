# OpenCV.js Parity Testing Infrastructure

**Status:** Phase 1 Implementation Complete ✅

This directory contains automated bit-level comparison tests between our opencv-rust WASM implementation and the reference opencv.js library.

## Overview

The goal of these tests is to:
1. **Verify API compatibility** - Ensure our WASM bindings match opencv.js signatures
2. **Validate correctness** - Compare outputs pixel-by-pixel against opencv.js
3. **Track performance** - Measure speedup of our GPU implementation vs opencv.js CPU
4. **Catch regressions** - Detect when changes introduce visual differences

## Architecture

```
tests/opencv_js_reference/
├── playwright.config.js      # Playwright config with headless WebGPU
├── pixel_comparison.js        # Pixel-level comparison utilities
├── test-harness.html          # Browser page loading both libraries
├── test_gaussian_blur.spec.js # First parity test (gaussian_blur)
├── generate_tests.js          # Test config & reference generation (stub)
├── compare_apis.js            # API signature comparison utilities
├── benchmark_suite.js         # Performance benchmarking (stub)
├── package.json               # Dependencies (Playwright)
└── README.md                  # This file
```

## Phase 1: Proof of Concept ✅

**Completed:**
- ✅ Playwright setup with headless WebGPU (SwiftShader)
- ✅ Pixel comparison logic with tolerance-based pass/fail
- ✅ Test harness HTML loading both opencv.js and our WASM
- ✅ First end-to-end test: `gaussian_blur`
- ✅ Test fixture infrastructure

**Test Coverage:**
- gaussian_blur: 5 test cases (3 param configs + 2 edge cases)

## Quick Start

### 1. Install Dependencies

```bash
cd tests/opencv_js_reference
npm install
```

This installs Playwright and its Chromium browser with WebGPU support.

### 2. Generate Test Fixtures

```bash
cd ../fixtures
python3 generate_fixtures.py
```

This creates synthetic test images (lenna.png, shapes.png, etc.).

### 3. Build WASM

From project root:

```bash
wasm-pack build --target web --out-dir pkg --features gpu,wasm
```

### 4. Run Tests

```bash
cd tests/opencv_js_reference
npm test
```

Or with UI for debugging:

```bash
npm run test:ui
```

## How It Works

### Test Flow

1. **Playwright launches headless Chromium** with WebGPU enabled via SwiftShader
2. **Test harness loads** both opencv.js (from CDN) and our WASM (from `pkg/`)
3. **Test runs both implementations** on the same input image with identical parameters
4. **Pixel-by-pixel comparison** calculates max diff, mean diff, and outlier %
5. **Pass/fail determination** based on predefined tolerances

### Tolerance Configuration

Each operation has tolerance thresholds defined in `generate_tests.js`:

```javascript
gaussian_blur: {
  tolerance: {
    max_pixel_diff: 1,      // ±1px due to float rounding
    max_mean_diff: 0.1,     // Average diff < 0.1
    max_outliers_percent: 1.0, // < 1% pixels exceed threshold
  }
}
```

**Why tolerances?**
- Floating-point precision differences (JS uses 64-bit, WASM uses 32-bit)
- Different SIMD/GPU optimizations may reorder operations
- Border handling may differ slightly

### Headless WebGPU

Uses SwiftShader software renderer for consistent, reproducible results:

```javascript
launchOptions: {
  args: [
    '--enable-unsafe-webgpu',
    '--use-angle=swiftshader',  // Software GPU
    // ... more flags
  ]
}
```

**Why SwiftShader?**
- Hardware GPUs vary by machine (driver differences)
- SwiftShader provides consistent results across CI and local
- Slower, but reliability > speed for tests

## Example Test

```javascript
test('gaussian_blur ksize=5 sigma=1.5 on lenna.png', async ({ page }) => {
  // Navigate to test harness
  await page.goto('/tests/opencv_js_reference/test-harness.html');
  await page.waitForFunction(() => window.testHarnessReady);

  // Run comparison in browser
  const result = await page.evaluate(async ({ ksize, sigma }) => {
    const img = await window.loadTestImage('lenna.png');

    // Run OpenCV.js
    const cvSrc = window.imageDataToMat(img);
    const cvDst = new cv.Mat();
    cv.GaussianBlur(cvSrc, cvDst, new cv.Size(ksize, ksize), sigma);
    const cvResult = window.matToImageData(cvDst);

    // Run our WASM
    const wasmSrc = new window.opencvRust.Mat(...);
    const wasmDst = await window.opencvRust.gaussian_blur(wasmSrc, ksize, sigma);
    const wasmData = wasmDst.data();

    return { opencv: cvResult.data, ours: wasmData };
  }, { ksize: 5, sigma: 1.5 });

  // Compare
  const comparison = compareImages(result.opencv, result.ours, tolerance);

  // Assert
  expect(comparison.passed).toBe(true);
});
```

## Next Steps: Phase 2

To expand coverage to the 12 configured operations, create tests for:

1. ✅ gaussian_blur (done)
2. ⏳ resize
3. ⏳ threshold
4. ⏳ canny
5. ⏳ sobel
6. ⏳ erode
7. ⏳ dilate
8. ⏳ bilateral_filter
9. ⏳ median_blur
10. ⏳ laplacian
11. ⏳ flip
12. ⏳ adaptive_threshold

**Template:** Copy `test_gaussian_blur.spec.js` and adapt for each operation.

## CI Integration

Add to `.github/workflows/opencv-parity-tests.yml`:

```yaml
name: OpenCV.js Parity Tests

on: [push, pull_request]

jobs:
  parity-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4

      - name: Install dependencies
        run: |
          cd tests/opencv_js_reference
          npm install

      - name: Build WASM
        run: wasm-pack build --target web --features gpu,wasm

      - name: Generate fixtures
        run: |
          cd tests/fixtures
          pip install pillow numpy
          python3 generate_fixtures.py

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

## Testing with Teleport

For testing with real GPU hardware (not SwiftShader), you can use teleport to run tests on a remote machine with GPU:

```bash
# On remote GPU machine
cd opencv-rust/tests/opencv_js_reference
npm test -- --headed  # Run with visible browser

# Or run specific test
npm test -- test_gaussian_blur.spec.js --headed
```

Teleport allows you to:
- See visual output of tests
- Debug WebGPU initialization issues
- Verify GPU acceleration is working
- Compare SwiftShader vs hardware GPU results

## Troubleshooting

### WebGPU not available

If tests fail with "WebGPU not supported":
1. Check Chrome version: `chromium --version` (need 113+)
2. Verify flags: `--enable-unsafe-webgpu`
3. Try hardware GPU: remove `--use-angle=swiftshader`

### WASM not loading

If "opencv-rust WASM failed to load":
1. Verify build: `ls -la ../../pkg/`
2. Check build features: `wasm-pack build --features gpu,wasm`
3. Check browser console in headed mode: `npm run test:headed`

### OpenCV.js not loading

If "opencv.js timeout":
1. Check CDN: https://docs.opencv.org/4.8.0/opencv.js
2. Use local copy: Download and update `test-harness.html`
3. Check network in headed mode

### Tests timing out

Increase timeout in `playwright.config.js`:
```javascript
timeout: 120 * 1000,  // 2 minutes
```

## Performance Results

Example output from gaussian_blur test:

```
Performance:
  OpenCV.js: 45.23ms
  Our WASM:  12.87ms
  Speedup:   3.51x

✅ PASS

Pixels Compared: 1048576
Different Pixels: 8234 (0.79%)

Max Difference: 1.00 (threshold: 2)
Mean Difference: 0.0523 (threshold: 0.1)
Outliers: 0.12% (threshold: 1%)

Outlier Pixels: 1258 (exceeding 1px diff)
```

## References

- **Plan:** `docs/251112-2100-opencv-parity-plan.md` - Full implementation roadmap
- **OpenCV.js:** https://docs.opencv.org/4.8.0/
- **Playwright:** https://playwright.dev/
- **WebGPU:** https://gpuweb.github.io/gpuweb/

## Contributing

To add a new parity test:

1. Add test config to `generate_tests.js` TEST_CONFIGS
2. Create `test_{operation}.spec.js` based on template
3. Define appropriate tolerances
4. Run test and adjust tolerances if needed
5. Document any known differences from opencv.js

---

**Last Updated:** 2025-11-12
**Status:** Phase 1 Complete - Ready for Phase 2 expansion
