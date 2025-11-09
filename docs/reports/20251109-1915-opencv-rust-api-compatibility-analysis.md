# OpenCV-Rust API Compatibility Analysis

**Date**: 2025-11-09 19:15
**Objective**: Ensure drop-in replacement compatibility with opencv-rust bindings

## Executive Summary

This analysis compares our pure Rust OpenCV implementation with the official opencv-rust bindings (docs.rs/opencv) to identify API differences and ensure drop-in replacement compatibility.

### Key Findings

**Status**: ⚠️ Partial Compatibility - Significant API gaps identified

- **Function naming**: ✅ Mostly compatible (snake_case conventions)
- **Mat API**: ⚠️ Missing critical methods
- **imgproc functions**: ✅ Good compatibility
- **Type conversions**: ⚠️ Needs review

## Detailed Analysis

### 1. Mat Struct API Comparison

#### Missing Critical Methods

Our implementation is missing these standard opencv-rust Mat methods:

**Creation Methods:**
- `zeros()`, `zeros_size()`, `zeros_nd()` - Create zero-filled matrices
- `ones()`, `ones_size()`, `ones_nd()` - Create one-filled matrices
- `eye()`, `eye_size()` - Create identity matrices
- `diag_mat()` - Create diagonal matrix
- `new_rows_cols()` - Standard constructor (we use `new()`)
- `new_size()` - Constructor from Size
- `new_nd()`, `new_nd_vec()` - N-dimensional array creation

**Data Access Methods:**
- `from_slice()`, `from_slice_mut()` - Create from slice data
- `from_slice_2d()` - Create from 2D slice
- `from_bytes()`, `from_bytes_mut()` - Create from byte slice
- `from_exact_iter()` - Create from iterator
- `copy()`, `copy_mut()` - Copy/reference matrices

**Property/Info Methods:**
- `type_()` - Complete type information
- `is_continuous()` - Check if data is contiguous
- `elem_size()` - Element size in bytes
- `step` - Bytes per row accessor
- `total()` - Total element count
- `dims` - Number of dimensions

**ROI/Slicing Methods:**
- `roi_mut()` - Mutable ROI (we only have immutable `roi()`)
- `roi_2_mut()` - Extract two non-overlapping ROIs
- `rowscols()`, `rowscols_mut()` - Row/column range extraction
- `rowscols_def()`, `rowscols_def_mut()` - Row ranges with default columns
- `ranges()`, `ranges_mut()` - Multi-dimensional ranges

#### Existing Methods - Naming Differences

| Our Method | opencv-rust Method | Action Required |
|------------|-------------------|-----------------|
| `new()` | `new_rows_cols()` | Rename or add alias |
| `new_with_default()` | `new_rows_cols_with_default()` | Rename or add alias |
| `from_raw()` | Custom (not standard) | Keep as extension |
| `clone_mat()` | `clone()` (trait) | Use standard Clone trait |
| `to_array3()` | Not in opencv-rust | Keep as extension |
| `from_array3()` | Not in opencv-rust | Keep as extension |

#### Methods We Have That Match ✅

- `rows()` - ✅ Matches
- `cols()` - ✅ Matches
- `channels()` - ✅ Matches
- `depth()` - ✅ Matches (returns enum)
- `is_empty()` - ✅ Matches
- `roi()` - ✅ Matches (but need mutable variant)
- `size()` - ✅ Matches
- `data()`, `data_mut()` - ✅ Matches
- `at()`, `at_mut()` - ✅ Matches
- `at_unchecked()`, `at_mut_unchecked()` - ✅ Matches
- `set_to()` - ✅ Matches

### 2. imgproc Module Function Comparison

#### Functions We Have That Match ✅

| Function | Status | Notes |
|----------|--------|-------|
| `cvt_color()` | ✅ | Correct name |
| `gaussian_blur()` | ✅ | Correct name |
| `blur()` | ✅ | Correct name |
| `median_blur()` | ✅ | Correct name |
| `bilateral_filter()` | ✅ | Correct name |
| `resize()` | ✅ | Correct name |
| `threshold()` | ✅ | Correct name |
| `adaptive_threshold()` | ✅ | Correct name |
| `canny()` | ✅ | Correct name |
| `sobel()` | ✅ | Correct name |
| `laplacian()` | ✅ | Correct name |
| `scharr()` | ✅ | Correct name |
| `erode()` | ✅ | Correct name |
| `dilate()` | ✅ | Correct name |
| `morphology_ex()` | ✅ | Correct name |
| `find_contours()` | ✅ | Correct name |
| `contour_area()` | ✅ | Correct name |
| `arc_length()` | ✅ | Correct name |
| `approx_poly_dp()` | ✅ | Correct name |
| `bounding_rect()` | ✅ | Correct name |
| `line()` | ✅ | Correct name |
| `rectangle()` | ✅ | Correct name |
| `circle()` | ✅ | Correct name |
| `ellipse()` | ✅ | Correct name |
| `polylines()` | ✅ | Correct name |
| `fill_poly()` | ✅ | Correct name |
| `put_text()` | ✅ | Correct name |
| `hough_lines()` | ✅ | Correct name |
| `hough_lines_p()` | ✅ | Correct name |
| `hough_circles()` | ✅ | Correct name |
| `calc_hist()` | ✅ | Correct name |
| `equalize_hist()` | ✅ | Correct name |
| `compare_hist()` | ✅ | Correct name |
| `calc_back_project()` | ✅ | Correct name |
| `warp_affine()` | ✅ | Correct name |
| `warp_perspective()` | ✅ | Correct name |
| `get_rotation_matrix_2d()` | ✅ | Correct name |
| `get_affine_transform()` | ✅ | Correct name |
| `rotate()` | ✅ | Correct name |
| `flip()` | ✅ | Correct name |
| `distance_transform()` | ✅ | Correct name |
| `watershed()` | ✅ | Correct name |

#### Additional Functions in Our Implementation

These are extensions not in standard opencv-rust:
- `circle_filled()` - Extension of circle
- `guided_filter()` - Advanced filter
- `gabor_filter()` - Advanced filter
- `laplacian_of_gaussian()` - Advanced filter
- `non_local_means_denoising()` - Advanced filter
- `anisotropic_diffusion()` - Advanced filter
- `normalize_hist()` - Helper function
- `moments()` - Contour moments

### 3. Type System Comparison

#### Enum Names

| Our Type | opencv-rust Type | Status |
|----------|-----------------|--------|
| `MatDepth` | Uses CV_* type system | ⚠️ Different approach |
| `ColorConversionCode` | Uses COLOR_* constants | ⚠️ Different approach |
| `InterpolationFlag` | `InterpolationFlags` | ⚠️ Different name |
| `ThresholdType` | `ThresholdTypes` | ⚠️ Different name |

**Note**: opencv-rust uses integer constants (e.g., `COLOR_BGR2GRAY`) while we use Rust enums (e.g., `ColorConversionCode::BgrToGray`). Our approach is more Rust-idiomatic but less compatible.

### 4. Function Signature Differences

Need to verify these match exactly:

```rust
// opencv-rust
pub fn gaussian_blur(
    src: &dyn ToInputArray,
    dst: &mut dyn ToOutputArray,
    ksize: Size,
    sigma_x: f64,
    sigma_y: f64,
    border_type: i32
) -> Result<()>

// Our implementation
pub fn gaussian_blur(
    src: &Mat,
    dst: &mut Mat,
    ksize: Size,
    sigma_x: f64
) -> Result<()>
```

⚠️ **Issue**: We're missing optional parameters that opencv-rust supports.

## Recommendations

### High Priority (Required for Drop-in Compatibility)

1. **Add Mat factory methods**: `zeros()`, `ones()`, `eye()`
2. **Rename Mat::new**: Add `new_rows_cols()` as primary constructor
3. **Add Mat property methods**: `type_()`, `elem_size()`, `step`, `total()`, `dims`
4. **Add mutable ROI methods**: `roi_mut()`, `rowscols_mut()`, etc.
5. **Review function signatures**: Add missing optional parameters

### Medium Priority (Improves Compatibility)

1. **Add data construction methods**: `from_slice()`, `from_bytes()`, etc.
2. **Add copy methods**: `copy()`, `copy_mut()`
3. **Consider type system alignment**: Option to use integer constants instead of enums

### Low Priority (Nice to Have)

1. Keep our Rust-idiomatic extensions as additional features
2. Add `#[deprecated]` warnings for methods with non-standard names
3. Create compatibility layer/facade for complete drop-in replacement

## Implementation Plan

### Phase 1: Core Mat API (Immediate)
- Add `zeros()`, `ones()`, `eye()` factory methods
- Add `new_rows_cols()` and deprecate/alias `new()`
- Add `type_()`, `elem_size()`, `step`, `total()`, `dims` properties
- Add `is_continuous()` method

### Phase 2: Mat Data Methods
- Add `from_slice()`, `from_slice_mut()`, `from_bytes()`
- Add `copy()`, `copy_mut()` methods
- Add `roi_mut()` and other mutable ROI methods

### Phase 3: Function Signature Updates
- Review all imgproc functions for parameter completeness
- Add missing optional parameters
- Ensure return types match exactly

### Phase 4: Type System Review
- Decide on enum vs constant approach
- Consider providing both for maximum compatibility
- Document any intentional differences

## Testing Strategy

1. **API compatibility tests**: Write tests that compile against both implementations
2. **Migration guide**: Document exact differences for users
3. **Benchmark comparisons**: Ensure performance parity
4. **Integration tests**: Test drop-in replacement scenarios

## Conclusion

Our implementation has excellent imgproc function naming compatibility but significant gaps in the Mat API. The main blocker for drop-in compatibility is:

1. Missing Mat factory methods (zeros, ones, eye)
2. Constructor naming differences (new vs new_rows_cols)
3. Missing property accessors (type_, elem_size, step, total, dims)
4. Missing mutable ROI/slicing methods

These gaps must be addressed to achieve true drop-in replacement compatibility with opencv-rust bindings.
