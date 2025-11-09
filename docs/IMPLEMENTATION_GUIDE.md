# OpenCV-Rust Implementation Guide

## Overview

This guide documents the systematic approach for implementing all 102 OpenCV features with both CPU and WebGPU acceleration, following test-driven development principles.

## Current Status

- **Total Features**: 102
- **Implemented**: 4 (gaussian_blur, canny, resize, threshold)
- **Remaining**: 98
- **Completion**: 3.9%

## Implementation Pattern

### Code Structure

Each feature follows this consistent pattern:

```
opencv-rust/
├── src/
│   ├── imgproc/        # Image processing operations
│   │   ├── filter.rs    # Filtering operations
│   │   ├── edge.rs      # Edge detection
│   │   ├── geometric.rs # Geometric transforms
│   │   └── ...
│   └── gpu/
│       └── ops/         # GPU implementations
│           ├── blur.rs
│           ├── canny.rs
│           └── ...
├── tests/
│   ├── test_accuracy_blur.rs
│   ├── test_accuracy_canny.rs
│   └── ...
└── examples/
    └── web-benchmark/
        └── src/demos/demoRegistry.js
```

### Implementation Stages

#### Stage 1: Test-First Development
1. **Migrate OpenCV Tests**
   - Find equivalent OpenCV C++ test
   - Translate to Rust using our Mat API
   - Create accuracy tests in `tests/test_accuracy_<feature>.rs`
   - Include edge cases, boundary conditions, and visual inspection tests

2. **Test Categories**
   ```rust
   #[test]
   fn test_<feature>_deterministic() { /* Same input = same output */ }

   #[test]
   fn test_<feature>_uniform_image() { /* Validate on uniform input */ }

   #[test]
   fn test_<feature>_boundary() { /* Border handling */ }

   #[test]
   fn test_<feature>_multichannel() { /* RGB, RGBA support */ }

   #[test]
   fn test_<feature>_output_range() { /* Valid value ranges */ }

   #[test]
   #[ignore]
   fn test_<feature>_visual_inspection() { /* Print for manual review */ }
   ```

#### Stage 2: CPU Implementation
1. **Create Function Stub**
   ```rust
   pub fn <feature>(src: &Mat, dst: &mut Mat, params...) -> Result<()> {
       // Parameter validation
       if src.depth() != MatDepth::U8 {
           return Err(Error::UnsupportedOperation(...));
       }

       // Try GPU first (if available)
       #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
       {
           if crate::gpu::gpu_available() {
               if let Ok(()) = crate::gpu::ops::<feature>_gpu(src, dst, params) {
                   return Ok(());
               }
           }
       }

       // Fallback to CPU
       <feature>_cpu(src, dst, params)
   }
   ```

2. **Implement CPU Algorithm**
   - Follow OpenCV's algorithm faithfully
   - Use Rust best practices (iterators, pattern matching)
   - Optimize with rayon for parallelization where appropriate
   - Document algorithm details and complexity

3. **Run Tests**
   ```bash
   cargo test test_accuracy_<feature>
   ```

#### Stage 3: GPU Implementation
1. **Create WGSL Shader**
   - Place in `src/gpu/shaders/<feature>.wgsl`
   - Implement compute shader following WebGPU best practices
   - Use workgroup shared memory for optimization
   - Document shader parameters and algorithm

2. **Create GPU Operation**
   ```rust
   // src/gpu/ops/<feature>.rs

   #[repr(C)]
   #[derive(Copy, Clone, Pod, Zeroable)]
   struct <Feature>Params {
       width: u32,
       height: u32,
       channels: u32,
       // feature-specific params
       _pad: [u32; N], // Ensure 16-byte alignment
   }

   pub async fn <feature>_gpu_async(src: &Mat, dst: &mut Mat, params...) -> Result<()> {
       // Async GPU implementation
   }

   #[cfg(not(target_arch = "wasm32"))]
   pub fn <feature>_gpu(src: &Mat, dst: &mut Mat, params...) -> Result<()> {
       pollster::block_on(<feature>_gpu_async(src, dst, params))
   }
   ```

3. **Test GPU vs CPU**
   ```bash
   cargo test --features gpu test_gpu
   ```

#### Stage 4: Integration
1. **Update Module Exports**
   - Add to `src/imgproc/mod.rs`
   - Add to `src/gpu/ops/mod.rs`

2. **Update WASM Bindings**
   ```rust
   // src/wasm/mod.rs
   #[wasm_bindgen]
   pub async fn <feature>(mat: &WasmMat, params...) -> Result<WasmMat, JsValue> {
       // WASM wrapper
   }
   ```

3. **Update Demo Registry**
   - Mark as `implemented: true` in `examples/web-benchmark/src/demos/demoRegistry.js`
   - Mark as `gpuAccelerated: true` if GPU version exists

## Priority Order (From plan.md)

### P0 - Critical (Must Have) - IMPLEMENT FIRST
1. ✅ **Gaussian Blur** - DONE
2. ✅ **Resize** - DONE
3. ✅ **Canny Edge Detection** - DONE
4. ✅ **Threshold** - DONE
5. ⏳ **Convert Color (RGB to Gray)** - Next
6. ⏳ **Sobel**
7. ⏳ **Drawing Functions** (Line, Rectangle, Circle)
8. ⏳ **Contour Detection**
9. ⏳ **Feature Detection** (SIFT/ORB)

### P1 - Important (Should Have)
10. Median Blur
11. Bilateral Filter
12. Adaptive Threshold
13. Histogram Equalization
14. Morphology Operations (Erode, Dilate, Opening, Closing)
15. Hough Lines
16. ArUco Detection
17. Background Subtraction

### P2 - Nice to Have (Could Have)
- Advanced Filters (Guided, Gabor, LoG, NLM Denoising)
- Optical Flow
- Object Tracking
- Camera Calibration
- HDR & Tone Mapping
- Super Resolution
- Panorama Stitching

### P3 - Future (Won't Have Initially)
- DNN Integration
- Advanced ML Models
- Stereo Vision
- Advanced Shape Analysis

## Algorithm Faithfulness

### OpenCV Reference Sources
When implementing, consult these OpenCV sources in order:
1. **OpenCV C++ Source**: https://github.com/opencv/opencv/tree/4.x/modules
2. **OpenCV Documentation**: https://docs.opencv.org/4.x/
3. **OpenCV Test Suite**: For edge cases and expected behavior

### Key Principles
1. **Match Algorithm Behavior**: Reproduce OpenCV's output exactly where possible
2. **Match API Design**: Keep function signatures similar to OpenCV
3. **Preserve Performance Characteristics**: O(n) algorithms stay O(n)
4. **Document Differences**: If deviating, document why

### Acceptable Rust Libraries

Use pure-Rust libraries with minimal dependencies:

✅ **Allowed**:
- `rayon` - Parallelization
- `wgpu` - GPU compute
- `bytemuck` - Safe byte casting
- `nalgebra` - Linear algebra (for transforms, calibration)
- `image` - Image codecs only
- `rustfft` - FFT operations
- `rand` - Random number generation

❌ **Not Allowed**:
- Bindings to C/C++ libraries (opencv-rust bindings, etc.)
- Heavy dependencies with deep dependency trees
- Libraries that are just wrappers around foreign code

### Performance Targets

| Operation Type | CPU Target | GPU Target | GPU Speedup |
|---------------|-----------|-----------|-------------|
| Simple Filters | < 50ms | < 5ms | > 10x |
| Edge Detection | < 100ms | < 10ms | > 10x |
| Complex Features | < 500ms | < 50ms | > 10x |
| ML Operations | < 2s | < 200ms | > 10x |

## Testing Approach

### Test-Driven Development Workflow

```bash
# 1. Create test file first
touch tests/test_accuracy_<feature>.rs

# 2. Write tests (they will fail)
cargo test test_accuracy_<feature> --lib

# 3. Implement CPU version
# Edit src/imgproc/<module>.rs

# 4. Pass CPU tests
cargo test test_accuracy_<feature>

# 5. Implement GPU version
# Create src/gpu/ops/<feature>.rs
# Create src/gpu/shaders/<feature>.wgsl

# 6. Pass GPU tests
cargo test --features gpu test_accuracy_<feature>

# 7. Verify WASM builds
wasm-pack build --target web --features gpu,wasm
```

### Test Data Sources

1. **Synthetic Data**: Generate in tests
2. **OpenCV Test Data**: Port from opencv/modules/*/test/
3. **Real Images**: Small test images in tests/data/

### Accuracy Validation

For each feature, validate against OpenCV:
```python
# Generate expected output with OpenCV
import cv2
import numpy as np

img = cv2.imread('test.jpg')
result = cv2.<feature>(img, params)
np.save('expected_output.npy', result)
```

Then in Rust tests:
```rust
let expected = load_npy("expected_output.npy");
let actual = <feature>(&src, params);
assert_images_within_tolerance(&expected, &actual, tolerance);
```

## Implementation Checklist

For each feature:
- [ ] Create test file `tests/test_accuracy_<feature>.rs`
- [ ] Migrate OpenCV tests (minimum 8 test cases)
- [ ] Implement CPU version in `src/<module>/<file>.rs`
- [ ] Pass all CPU tests
- [ ] Create GPU shader `src/gpu/shaders/<feature>.wgsl`
- [ ] Implement GPU version in `src/gpu/ops/<feature>.rs`
- [ ] Pass GPU tests
- [ ] Create WASM binding in `src/wasm/mod.rs`
- [ ] Update demo registry (mark implemented)
- [ ] Verify WASM build succeeds
- [ ] Benchmark CPU vs GPU performance
- [ ] Document in function docstring
- [ ] Update progress in docs/plan.md

## Progress Tracking

Track implementation in `docs/IMPLEMENTATION_STATUS.md`:

```markdown
## Feature Implementation Status

### Image Filtering (11 features)
- [x] Gaussian Blur (CPU✓ GPU✓ WASM✓)
- [ ] Box Blur (CPU_ GPU_ WASM_)
- [ ] Median Blur (CPU_ GPU_ WASM_)
...
```

## Example: Implementing Sobel

### 1. Create Test (test_accuracy_sobel.rs)
```rust
use opencv_rust::imgproc::sobel;
use test_utils::*;

#[test]
fn test_sobel_x_gradient() {
    // Create vertical edge
    let mut src = Mat::new(10, 10, 1, MatDepth::U8).unwrap();
    for row in 0..10 {
        for col in 0..10 {
            src.at_mut(row, col).unwrap()[0] = if col < 5 { 0 } else { 255 };
        }
    }

    let mut dst = Mat::new(1, 1, 1, MatDepth::U8).unwrap();
    sobel(&src, &mut dst, 1, 0, 3).unwrap();

    // Strong response in center column
    assert!(dst.at(5, 5).unwrap()[0] > 200);
}

// 7 more tests...
```

### 2. Implement CPU Version
```rust
// src/imgproc/edge.rs
pub fn sobel(src: &Mat, dst: &mut Mat, dx: i32, dy: i32, ksize: i32) -> Result<()> {
    // Validate params
    if dx < 0 || dy < 0 || dx + dy == 0 {
        return Err(Error::InvalidParameter("Invalid derivative order".into()));
    }

    // GPU path...

    // CPU implementation
    sobel_cpu(src, dst, dx, dy, ksize)
}

fn sobel_cpu(src: &Mat, dst: &mut Mat, dx: i32, dy: i32, ksize: i32) -> Result<()> {
    // Implement Sobel algorithm following OpenCV
    let kernel = get_sobel_kernel(ksize, dx, dy)?;
    apply_filter(src, dst, &kernel)
}
```

### 3. Implement GPU Version
```wgsl
// src/gpu/shaders/sobel.wgsl
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: SobelParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Sobel kernel computation
}
```

```rust
// src/gpu/ops/sobel.rs
pub async fn sobel_gpu_async(src: &Mat, dst: &mut Mat, dx: i32, dy: i32, ksize: i32) -> Result<()> {
    // GPU implementation
}
```

## Next Steps

1. Review and approve this implementation guide
2. Start with P0 feature #5: Convert Color (RGB to Gray)
3. Follow TDD: Tests → CPU → GPU → Integration
4. Maintain momentum: Aim for 2-3 features per day
5. Update progress tracking after each feature

## Resources

- OpenCV Source: https://github.com/opencv/opencv
- OpenCV Docs: https://docs.opencv.org/4.x/
- WebGPU Spec: https://www.w3.org/TR/webgpu/
- WGSL Spec: https://www.w3.org/TR/WGSL/
- This Project: https://github.com/boxabirds/opencv-rust

---

**Goal**: Complete all 102 features with CPU, GPU, and WASM support, tested against OpenCV reference implementation.
