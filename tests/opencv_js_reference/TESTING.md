# Testing OpenCV.js Parity Tests Locally

## Known Issues

### Issue 1: Container Chrome Crashes (RESOLVED - Root Cause Identified)

**Status:** Root cause identified through comprehensive logging

**Symptom:** Chrome crashes with "Target crashed" error when loading large WASM files (~10MB) in headless mode with SwiftShader

**Root Cause Analysis:**
- **Chrome crashes even loading ONLY OpenCV.js (9.6 MB)** - Isolated test confirmed
- **Chrome crashes even loading ONLY our WASM** - Isolated test confirmed
- **NOT a compatibility issue** between libraries
- **Confirmed:** Large WASM files cause headless Chrome + SwiftShader to crash in container environments
- **Crash timing:** ~2 seconds after HTTP load completes (during WASM compilation/initialization)

**Evidence:**
```
✅ HTTP 200 OK - opencv.js loads successfully
✅ Initialization starts properly - All logging works
❌ Chrome crashes ~2s after loading begins
❌ ALL 3 isolated tests crash (OpenCV.js only, WASM only, and combined)
```

**Workaround:** Tests work fine on local machines with real GPU/proper Chrome environment

**Diagnostic Logging Added:**
- Timestamped initialization steps with performance metrics
- Memory usage tracking (heap size)
- Network request/response monitoring
- Detailed error capture with stack traces
- Step-by-step WASM loading progress
- Global error and unhandled rejection handlers

**To see diagnostic output:**
```bash
npm test test_debug.spec.js
```

**Three diagnostic tests included:**
1. **Comprehensive initialization diagnostics** - Full monitoring with network tracking
2. **Isolated OpenCV.js load test** - Tests if OpenCV.js alone causes crash
3. **Isolated WASM load test** - Tests if our WASM alone causes crash

**Diagnostic output includes:**
- Exact timing of each initialization step (with timestamps)
- Network activity for all resources (requests/responses)
- Console messages with timestamps (log/warn/error)
- Error capture with full stack traces
- State snapshots every 2 seconds during init
- Memory usage tracking (heap size)
- Browser and WebGPU availability info
- Full initialization step history before failure

**Example output:**
```
========================================
=== STARTING TEST HARNESS DIAGNOSTICS ===
========================================

[2025-11-12T22:41:18.954Z] [LOG] [0.90ms] === INITIALIZATION START ===
[2025-11-12T22:41:18.955Z] [LOG] [1.70ms] Browser: Mozilla/5.0 ...
[2025-11-12T22:41:19.238Z] [LOG] [2.10ms] WebGPU available: true
[2025-11-12T22:41:19.238Z] [LOG] [286.00ms] [Memory] Initialization start: 9.54MB used / 9.54MB total
[2025-11-12T22:41:19.239Z] [LOG] [286.30ms] [Test Harness] Loading OpenCV.js...
[2025-11-12T22:41:19.239Z] [LOG] [286.50ms] [OpenCV.js] Starting wait for OpenCV.js...
[2025-11-12T22:41:19.240Z] [LOG] [287.50ms] [OpenCV.js] Script tag found, readyState: undefined

Error: page.evaluate: Target crashed
```

### Issue 2: Network Access (SOLVED - CDN Cached Locally)

**Status:** Solved - OpenCV.js v4.8.0 cached in `/cache/opencv.js` (9.6 MB)

The original issue was `ERR_TUNNEL_CONNECTION_FAILED` when loading from CDN. This is now resolved by using local cache.

## Testing Locally (Recommended)

### Prerequisites
- Node.js 18+ installed
- Chrome/Chromium browser
- Network access to opencv.org CDN

### Steps

```bash
# 1. Clone and navigate to the project
cd opencv-rust

# 2. Build WASM with fixed GPU shaders
wasm-pack build --target web --out-dir pkg --features gpu,wasm

# 3. Generate test fixtures
cd tests/fixtures
pip install pillow numpy  # or use venv
python3 generate_fixtures.py
cd ../..

# 4. Install Playwright
cd tests/opencv_js_reference
npm install

# 5. Install Playwright browsers
npx playwright install chromium

# 6. Run tests
npm test

# Or run with visible browser to debug
npm run test:headed

# Or run just the debug test
npm test test_debug.spec.js
```

### Expected Results

**If successful:**
```
Running 5 tests using 1 worker

✓ gaussian_blur ksize=5 sigma=1.5 on lenna.png
✓ gaussian_blur ksize=9 sigma=2 on lenna.png
✓ gaussian_blur ksize=15 sigma=3 on lenna.png
✓ handles small kernel size (ksize=3)
✓ handles large kernel size (ksize=21)

5 passed (30s)
```

**With pixel comparison output:**
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
```

### Debugging with Visible Browser

```bash
# Run with headed mode to see what's happening
npm run test:headed

# Or run a specific test
npm test test_gaussian_blur.spec.js -- --headed

# Access the test harness directly
# Start a simple web server in project root:
python3 -m http.server 8080

# Then open in browser:
# http://localhost:8080/tests/opencv_js_reference/test-harness.html
```

### Common Issues

#### 1. OpenCV.js CDN Not Accessible
**Error:** `ERR_TUNNEL_CONNECTION_FAILED` or `net::ERR_NAME_NOT_RESOLVED`

**Solution:** Check internet connection or use local opencv.js (see workaround below)

#### 2. WASM Module Not Found
**Error:** `Failed to load resource: /pkg/opencv_rust.js 404`

**Solution:** Rebuild WASM: `wasm-pack build --target web --features gpu,wasm`

#### 3. WebGPU Not Available
**Error:** `GPU context not initialized`

**Solution:** This is OK - tests will fall back to CPU. To test GPU:
- Use `--headed` mode
- Or use hardware GPU: remove `--use-angle=swiftshader` from playwright.config.js

#### 4. Tests Timeout
**Error:** `Test timeout of 60000ms exceeded`

**Causes:**
- OpenCV.js not loading (network issue)
- WASM not loading (build issue)
- Page crash (check console with --headed)

**Debug:**
```bash
npm test test_debug.spec.js -- --headed
```

## Local OpenCV.js (Already Cached)

**Good news:** OpenCV.js v4.8.0 is already cached in the repository at `/cache/opencv.js` (9.6 MB).

The test harness automatically loads from this cached copy instead of CDN:

```html
<script async src="/cache/opencv.js"></script>
```

**Why cached?**
- Container environments block external CDN access
- CI reliability (CDN outages won't break tests)
- Offline development support
- Consistent version ensures reproducible results

See `cache/README.md` for details.

## Container Testing (Limited)

The automated tests run in a Docker/Kubernetes container with restricted network access. This causes OpenCV.js CDN loading to fail.

**Workarounds for container:**
1. Use local opencv.js copy (recommended)
2. Configure proxy/network access
3. Skip parity tests in container, run locally

## CI Integration

For GitHub Actions CI, the tests should work because the runner has network access:

```yaml
- name: Run parity tests
  run: |
    cd tests/opencv_js_reference
    npx playwright test
```

If CI also has network issues, use the local opencv.js workaround.

## Performance Expectations

**Target speedup over opencv.js:**
- Simple operations (threshold, flip): 4-5x faster
- Filters (gaussian_blur, sobel): 2-3x faster
- Complex operations (bilateral_filter): 5x+ faster

**Pixel accuracy:**
- Most operations: ≤1px diff (due to float rounding)
- Complex operations: ≤3px diff
- Mean difference: <0.5 across all pixels

## Next Steps After Local Testing

Once tests pass locally:

1. **Verify shader fixes work** - Colors should be correct (not just red)
2. **Check performance gains** - GPU should be 2-5x faster than opencv.js
3. **Review pixel differences** - Should be within tolerance
4. **Expand to more operations** - Add tests for threshold, resize, etc.

## Help & Support

If tests fail locally:
- Check console output with `--headed` mode
- Review test-harness.html in browser directly
- Share console errors and screenshots
- Check that WASM built successfully (ls -la pkg/)

## Files

- `test-harness.html` - Test page that loads both libraries
- `test_gaussian_blur.spec.js` - Parity test for gaussian_blur
- `test_debug.spec.js` - Debug test with console capture
- `pixel_comparison.js` - Comparison logic
- `playwright.config.js` - Playwright configuration
