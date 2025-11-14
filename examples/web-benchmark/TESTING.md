# Testing Guide for Web Benchmark Operations

## What Was Fixed

### 1. Distance Transform Type Mismatch ✅
**Issue**: GPU output was F32 but WASM wrapper expected U8
**Fix**: Changed WASM wrapper to accept F32 and normalize to U8 for display
**File**: `src/wasm/misc/various.rs:25-67`

### 2. Comprehensive Error Logging ✅
**Added**:
- Detailed console logging for all operations
- Parameter logging
- Error stack traces
- WASM-specific error detection
- User-friendly alerts with instructions
- Operation-specific debugging info

**Files Changed**:
- `src/App.jsx:175-327` - Main error handling
- `src/App.jsx:329-934` - Individual operation logging

## How to Test

### 1. Open Browser Dev Tools
Press `F12` or right-click → Inspect to open the console

### 2. Load the App
Navigate to: http://localhost:3000

### 3. Upload a Test Image
- Click "Choose File" or drag & drop
- Use any image (JPG, PNG, etc.)

### 4. Test Each Operation
Click through the gallery on the left side and test each operation.

## What to Look For

### Success Case
Console output will show:
```
================================================================================
[operation_name] "Operation Name" - Starting processing
Parameters: {...}
================================================================================
[operation_name] Input image: 640x480, 1228800 bytes
[operation_name] Creating WasmMat...
[operation_name] WasmMat created: 640x480, 4 channels
[operation_name] ✓ Using GPU backend (WebGPU)
[operation_name] Warmup run (compiling GPU shaders)...
[operation_name] Warmup result: 640x480
[operation_name] Starting timed GPU run...
[operation_name] ✓ GPU processing complete in 2.45ms
[operation_name] GPU result: 640x480, 4 channels
[operation_name] Converted to data URL: 12345 chars
[operation_name] ✓ UI updated successfully
[operation_name] ================================================================================
[operation_name] ✓ Processing complete
```

### Failure Case
Console output will show:
```
================================================================================
[operation_name] ✗ OPERATION FAILED
================================================================================
Demo: Operation Name (operation_name)
Category: Filters
Parameters: {...}
Error type: Error
Error message: Detailed error message
Error stack: Full stack trace
================================================================================
```

**Alert will show**:
```
Operation "Operation Name" failed:

Error: [error message]

See browser console (F12) for full details including:
- Input parameters
- Error stack trace
- GPU/WASM debug logs
```

## Known Working Operations

Based on type checking, these operations should work:
- All basic filters (Gaussian Blur, Box Blur, Median Blur)
- All edge detection (Canny, Sobel, Scharr, Laplacian)
- All morphology operations (Erode, Dilate, Opening, Closing)
- All color conversions (RGB↔HSV, RGB↔LAB, RGB↔YCrCb)
- All arithmetic operations (Add, Subtract, Multiply, etc.)
- All bitwise operations (AND, OR, XOR, NOT)
- All geometric transforms (Resize, Flip, Rotate, Warp Affine)
- Histogram operations
- Threshold operations

## Operations to Check Carefully

These operations have complex implementations:
1. **Bilateral Filter** - Complex shader, check for parameter issues
2. **Distance Transform** - Just fixed, should now work ✅
3. **Hough Lines/Circles** - CPU fallback for Hough transform
4. **Feature Detection** (SIFT, ORB, etc.) - Complex algorithms
5. **Machine Learning** operations - May need training data
6. **Camera Calibration** - Needs specific input patterns
7. **Panorama Stitching** - Needs multiple images

## Debugging Tips

### If operation fails with "unreachable"
- The Rust code has an `unimplemented!()` or `unreachable!()` macro
- The operation isn't fully implemented
- Check console for "WASM unreachable instruction" message

### If operation fails with "RuntimeError"
- The Rust code panicked
- Check console for "WASM RuntimeError detected" message
- Look for parameter validation errors

### If operation returns null
- The WASM function returned None or null
- Check console for "GPU result is null/undefined"
- May indicate shader compilation failure or GPU context issues

### If GPU initialization fails
- Browser doesn't support WebGPU
- Use Chrome 113+ or Edge 113+
- Check that WebGPU is enabled in browser flags

## Reporting Issues

When reporting an issue, include:
1. **Operation name** (e.g., "Bilateral Filter")
2. **Full console output** (copy from console)
3. **Browser** (Chrome/Edge version)
4. **Parameters used** (shown in console log)
5. **Input image size** (shown in console log)

## Test Checklist

### Filters (10 operations)
- [ ] Gaussian Blur
- [ ] Box Blur
- [ ] Median Blur
- [ ] Bilateral Filter
- [ ] Guided Filter
- [ ] Gabor Filter
- [ ] NLM Denoising
- [ ] Anisotropic Diffusion
- [ ] Fast NLM
- [ ] Log Filter

### Edge Detection (4 operations)
- [ ] Canny
- [ ] Sobel
- [ ] Scharr
- [ ] Laplacian

### Morphology (7 operations)
- [ ] Erode
- [ ] Dilate
- [ ] Opening
- [ ] Closing
- [ ] Gradient
- [ ] Top Hat
- [ ] Black Hat

### Color Conversion (7 operations)
- [ ] To Grayscale
- [ ] To HSV
- [ ] To LAB
- [ ] To YCrCb
- [ ] HSV to RGB
- [ ] LAB to RGB
- [ ] YCrCb to RGB

### Geometric (6 operations)
- [ ] Resize
- [ ] Flip
- [ ] Rotate
- [ ] Warp Affine
- [ ] Warp Perspective
- [ ] Rotation Matrix 2D

### Thresholding (2 operations)
- [ ] Binary Threshold
- [ ] Adaptive Threshold

### Feature Detection (8 operations)
- [ ] Harris Corners
- [ ] Good Features to Track
- [ ] FAST
- [ ] SIFT
- [ ] ORB
- [ ] BRISK
- [ ] AKAZE
- [ ] KAZE

### Contours (9 operations)
- [ ] Find Contours
- [ ] Bounding Rectangle
- [ ] Contour Area
- [ ] Arc Length
- [ ] Approx Poly DP
- [ ] Moments
- [ ] Min Enclosing Circle
- [ ] Convex Hull
- [ ] Hu Moments

### Histogram (5 operations)
- [ ] Equalize Histogram
- [ ] Calc Histogram
- [ ] Normalize Histogram
- [ ] Compare Histograms
- [ ] Back Projection

### Drawing (5 operations)
- [ ] Draw Line
- [ ] Draw Rectangle
- [ ] Draw Circle
- [ ] Draw Ellipse
- [ ] Put Text

### Miscellaneous (39 remaining operations)
Test all other operations in each category.

---

**Total Operations**: 102
**Type-Safe Operations**: 101
**Fixed Operations**: 1 (Distance Transform)
