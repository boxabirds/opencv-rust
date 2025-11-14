# WASM Endpoint Test Status

## Test Infrastructure

- **Browser test suite**: `http://localhost:3000/test-suite.html`
- **Automated runner**: `node run-test-suite.js` (Puppeteer-based)
- **Total operations**: 102 endpoints across 15 categories

## Known Issues

### 1. Gabor Filter - Parameter Mismatch (FIXED)
**Status**: Fixed in test suite

**Problem**: Demo registry params (`frequency`, `orientation`, `sigma`) didn't match WASM function signature (`ksize`, `sigma`, `theta`, `lambda`, `gamma`, `psi`)

**Solution**: Added special case handling in `callDemoOperation()` to transform params correctly

### 2. WASM Panics Crash Page
**Status**: Architectural limitation

**Problem**: WASM "unreachable" instructions and panics crash the entire page, preventing test suite from continuing

**Impact**: Makes fully automated testing difficult - each crashing test stops the entire suite

**Workaround**: Test suite HTML can be run manually in browser to see results interactively

## Previously Fixed (from FIXES_APPLIED.md)

### ✅ RGBA Channel Handling (11 files, ~50 operations)
**Fixed**: Added runtime channel detection for 4-channel (RGBA) vs 3-channel (BGR) images

Operations now working:
- Edge Detection: Canny, Sobel, Scharr, Laplacian
- Feature Detection: SIFT, ORB, BRISK, AKAZE, KAZE, FAST, Harris
- Hough Transforms: Lines, Lines P, Circles
- Contours: All contour operations
- Segmentation: K-means, Watershed
- Histogram: All histogram operations
- Tracking: All trackers
- Thresholding: Adaptive threshold
- Distance Transform
- Laplacian of Gaussian (LoG)

**Files**:
1. `src/wasm/basic/edge.rs`
2. `src/wasm/basic/filtering.rs`
3. `src/wasm/basic/threshold.rs`
4. `src/wasm/calib3d/camera.rs`
5. `src/wasm/features/detection.rs`
6. `src/wasm/features/object.rs`
7. `src/wasm/imgproc/contour.rs`
8. `src/wasm/imgproc/histogram.rs`
9. `src/wasm/misc/various.rs`
10. `src/wasm/segmentation/cluster.rs`
11. `src/wasm/video/tracking.rs`

### ✅ Distance Transform Type Mismatch
**Fixed**: GPU outputs F32, added normalization to U8 for display

**File**: `src/wasm/misc/various.rs`

## Testing Strategy

### Manual Testing (Recommended)
1. Open `http://localhost:3000/test-suite.html`
2. Click "Start Tests"
3. Watch real-time progress
4. Download JSON results

**Advantages**:
- See exactly which operation crashes
- Get partial results even if some tests crash
- Can investigate crashes in DevTools

### Automated Testing (Limited)
```bash
node run-test-suite.js
```

**Limitations**:
- Will stop at first WASM panic/crash
- Cannot recover from "unreachable" errors
- Requires page reload between problematic tests

## Next Steps

To get complete test results:

1. **Run manual test suite** - Click through http://localhost:3000/test-suite.html
2. **Identify crashing operations** - Note which tests cause page crash
3. **Fix crashes** - Address WASM panics and unreachable instructions
4. **Re-run** - Test again after fixes

Or:

1. **Modify test suite** - Add page reload between tests to isolate crashes
2. **Skip known crashes** - Maintain blacklist of crashing operations
3. **Test in batches** - Run small groups of operations at a time

## Expected Pass Rate

Based on previous channel conversion fixes:
- **~50 operations** were fixed by RGBA handling
- **Total operations**: 102
- **Expected baseline**: 50-60% passing (assuming no other major issues)

Actual pass rate will depend on:
- GPU/CPU backend availability
- Missing WASM exports
- Unimplemented operations
- Additional parameter mismatches
- WASM panics/crashes
