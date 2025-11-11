# Backend Selection Migration Status

## Summary
**Completed: 23/102 operations (22.5%)**
**Remaining: 79/102 operations (77.5%)**

## Completed Operations (23)

### ✅ basic/edge.rs (4 operations) - ALL WITH GPU SUPPORT
- `canny_wasm` → `crate::gpu::ops::canny_gpu_async`
- `sobel_wasm` → `crate::gpu::ops::sobel_gpu_async`
- `scharr_wasm` → `crate::gpu::ops::scharr_gpu_async`
- `laplacian_wasm` → `crate::gpu::ops::laplacian_gpu_async`

### ✅ basic/filtering.rs (10 operations) - 5 WITH GPU, 5 CPU-ONLY
**With GPU:**
- `gaussian_blur_wasm` → `crate::gpu::ops::gaussian_blur_gpu_async`
- `blur_wasm` → `crate::gpu::ops::box_blur_gpu_async`
- `box_blur_wasm` → `crate::gpu::ops::box_blur_gpu_async`
- `median_blur_wasm` → `crate::gpu::ops::median_blur_gpu_async`
- `bilateral_filter_wasm` → `crate::gpu::ops::bilateral_filter_gpu_async`
- `filter2d_wasm` → `crate::gpu::ops::filter2d_gpu_async`

**CPU-only (future-proof pattern added):**
- `guided_filter_wasm`
- `gabor_filter_wasm`
- `nlm_denoising_wasm`
- `anisotropic_diffusion_wasm`
- `fast_nl_means_wasm`

### ✅ basic/threshold.rs (1 operation) - WITH GPU SUPPORT
- `adaptive_threshold_wasm` → `crate::gpu::ops::adaptive_threshold_gpu_async`

### ✅ imgproc/morphology.rs (9 operations) - 2 WITH GPU, 7 CPU-ONLY
**With GPU:**
- `erode_wasm` → `crate::gpu::ops::erode_gpu_async`
- `dilate_wasm` → `crate::gpu::ops::dilate_gpu_async`

**CPU-only (future-proof pattern added):**
- `morphology_opening_wasm`
- `morphology_closing_wasm`
- `morphology_gradient_wasm`
- `morphology_top_hat_wasm`
- `morphology_black_hat_wasm`
- `morphology_tophat_wasm` (alt casing)
- `morphology_blackhat_wasm` (alt casing)

## Git Commits
- Commit `a871ed2`: "Add backend selection to basic and morphology operations (23 funcs)"

## Remaining Operations (79)

### Priority 1: Operations with GPU Support (HIGH PRIORITY)

#### imgproc/geometric.rs (9 operations) - LIKELY 7+ WITH GPU
- `resize_wasm` → `crate::gpu::ops::resize_gpu_async` ✓
- `flip_wasm` → `crate::gpu::ops::flip_gpu_async` ✓
- `rotate_wasm` → `crate::gpu::ops::rotate_gpu_async` ✓
- `warp_affine_wasm` → `crate::gpu::ops::warp_affine_gpu_async` ✓
- `warp_perspective_wasm` → `crate::gpu::ops::warp_perspective_gpu_async` ✓
- `get_rotation_matrix_2d_wasm` - CPU only
- `remap_wasm` → `crate::gpu::ops::remap_gpu_async` ✓
- `pyrdown_wasm` → `crate::gpu::ops::pyrdown_gpu_async` ✓
- `pyrup_wasm` → `crate::gpu::ops::pyrup_gpu_async` ✓

#### imgproc/color.rs (11 operations) - LIKELY 10+ WITH GPU
- `cvt_color_*_wasm` variants (rgb_to_gray, rgb_to_hsv, hsv_to_rgb, etc.)
- All color conversion ops in src/gpu/ops/: rgb_to_gray, rgb_to_hsv, rgb_to_lab, rgb_to_ycrcb, hsv_to_rgb, lab_to_rgb, ycrcb_to_rgb ✓

#### arithmetic/ops.rs (9 operations) - ALL WITH GPU
- `add_wasm` → `crate::gpu::ops::add_gpu_async` ✓
- `subtract_wasm` → `crate::gpu::ops::subtract_gpu_async` ✓
- `multiply_wasm` → `crate::gpu::ops::multiply_gpu_async` ✓
- `add_weighted_wasm` → `crate::gpu::ops::add_weighted_gpu_async` ✓
- `absdiff_wasm` → `crate::gpu::ops::absdiff_gpu_async` ✓
- `min_wasm` → `crate::gpu::ops::min_gpu_async` ✓
- `max_wasm` → `crate::gpu::ops::max_gpu_async` ✓
- `convert_scale_wasm` → `crate::gpu::ops::convert_scale_gpu_async` ✓
- `normalize_wasm` → `crate::gpu::ops::normalize_gpu_async` ✓

#### comparison/bitwise.rs (9 operations) - ALL WITH GPU
- `bitwise_and_wasm` → `crate::gpu::ops::bitwise_and_gpu_async` ✓
- `bitwise_or_wasm` → `crate::gpu::ops::bitwise_or_gpu_async` ✓
- `bitwise_xor_wasm` → `crate::gpu::ops::bitwise_xor_gpu_async` ✓
- `bitwise_not_wasm` → `crate::gpu::ops::bitwise_not_gpu_async` ✓
- `in_range_wasm` → `crate::gpu::ops::in_range_gpu_async` ✓
- Plus other comparison operations

#### imgproc/histogram.rs (5 operations) - SOME WITH GPU
- `equalize_hist_wasm` → `crate::gpu::ops::equalize_hist_gpu_async` ✓
- `calc_hist_wasm` - likely CPU only
- `compare_hist_wasm` - likely CPU only
- Others - check src/gpu/ops/

### Priority 2: Operations Likely CPU-Only

#### imgproc/contour.rs (10 operations) - LIKELY CPU-ONLY
- `find_contours_wasm`
- `draw_contours_wasm`
- `contour_area_wasm`
- `arc_length_wasm`
- `approx_poly_dp_wasm`
- `convex_hull_wasm`
- `is_contour_convex_wasm`
- `bounding_rect_wasm`
- `min_area_rect_wasm`
- `min_enclosing_circle_wasm`

#### imgproc/drawing.rs (6 operations) - LIKELY CPU-ONLY
- `line_wasm`
- `rectangle_wasm`
- `circle_wasm`
- `ellipse_wasm`
- `polylines_wasm`
- `put_text_wasm`

#### features/detection.rs (8 operations) - LIKELY CPU-ONLY
- Feature detection operations (SIFT, SURF, ORB, etc.)

#### features/object.rs (4 operations) - LIKELY CPU-ONLY
- Object detection operations

#### ml/classifiers.rs (5 operations) - CPU-ONLY
- Machine learning classifier operations

#### video/tracking.rs (7 operations) - LIKELY CPU-ONLY
- Video tracking operations

#### calib3d/camera.rs (7 operations) - CPU-ONLY
- Camera calibration operations

#### dnn/network.rs (2 operations) - CPU-ONLY
- Deep neural network operations

#### segmentation/cluster.rs (2 operations) - LIKELY CPU-ONLY
- Segmentation and clustering operations

#### misc/various.rs (19 operations) - MIXED (CHECK EACH)
- Various utility operations
- Need to check src/gpu/ops/ for: split, merge, exp, log, sqrt, pow, lut, integral_image, distance_transform, gradient_magnitude

## Migration Pattern

### For operations WITH GPU support:

```rust
/// Operation description
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = operationName)]
pub async fn operation_wasm(src: &WasmMat, params...) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(...).map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Use backend selection
    match backend::get_backend() {
        1 => {
            // GPU path
            #[cfg(feature = "gpu")]
            {
                crate::gpu::ops::operation_gpu_async(&src.inner, &mut dst, params)
                    .await
                    .map_err(|e| JsValue::from_str(&format!("GPU error: {}. Try setBackend('auto') or setBackend('cpu')", e)))?;
                return Ok(WasmMat { inner: dst });
            }
            #[cfg(not(feature = "gpu"))]
            {
                return Err(JsValue::from_str("GPU not available in this build. Try setBackend('cpu')"));
            }
        }
        _ => {
            // CPU path
            crate::category::operation(&src.inner, &mut dst, params)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}
```

### For operations WITHOUT GPU support (future-proof):

```rust
/// Operation description
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(js_name = operationName)]
pub async fn operation_wasm(src: &WasmMat, params...) -> Result<WasmMat, JsValue> {
    let mut dst = Mat::new(...).map_err(|e| JsValue::from_str(&e.to_string()))?;

    // Use backend selection (CPU-only for now, future-proof for GPU)
    match backend::get_backend() {
        1 => {
            // GPU path not yet implemented
            #[cfg(feature = "gpu")]
            {
                return Err(JsValue::from_str("GPU operation not yet implemented. Try setBackend('cpu')"));
            }
            #[cfg(not(feature = "gpu"))]
            {
                return Err(JsValue::from_str("GPU not available in this build. Try setBackend('cpu')"));
            }
        }
        _ => {
            // CPU path
            crate::category::operation(&src.inner, &mut dst, params)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
        }
    }

    Ok(WasmMat { inner: dst })
}
```

## Migration Steps

### For Each File:

1. **Add backend import:**
   ```rust
   #[cfg(target_arch = "wasm32")]
   use crate::wasm::backend;
   ```

2. **Check for GPU implementations:**
   ```bash
   ls src/gpu/ops/{operation_name}.rs
   ```

3. **Update each operation function** with the appropriate pattern

4. **Test compilation:**
   ```bash
   cargo clippy --target wasm32-unknown-unknown --features wasm --no-deps
   ```

5. **Commit after each module:**
   ```bash
   git add src/wasm/{category}/{module}.rs
   git commit -m "Add backend selection to {module} operations ({count} funcs)"
   ```

## Recommended Order

1. **Phase 2: imgproc operations** (40 ops)
   - geometric.rs (9 ops - HIGH GPU support)
   - color.rs (11 ops - HIGH GPU support)
   - histogram.rs (5 ops - MIXED)
   - contour.rs (10 ops - CPU-only)
   - drawing.rs (6 ops - CPU-only)

2. **Phase 3: arithmetic and bitwise** (18 ops - ALL GPU)
   - arithmetic/ops.rs (9 ops)
   - comparison/bitwise.rs (9 ops)

3. **Phase 4: features** (12 ops - likely CPU-only)
   - features/detection.rs (8 ops)
   - features/object.rs (4 ops)

4. **Phase 5: remaining modules** (42 ops - mostly CPU-only)
   - ml/classifiers.rs (5 ops)
   - video/tracking.rs (7 ops)
   - calib3d/camera.rs (7 ops)
   - dnn/network.rs (2 ops)
   - segmentation/cluster.rs (2 ops)
   - misc/various.rs (19 ops - MIXED, check GPU ops)

## Compilation Status
✅ All 23 migrated operations compile successfully
✅ No errors in current codebase
✅ Ready for continued migration

## Next Steps

Continue with **Phase 2: geometric.rs** (highest GPU support ratio)
