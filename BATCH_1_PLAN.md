# Batch 1: Core Image Processing GPU Implementation

**Goal**: Implement GPU acceleration for 20 core image processing features
**Target**: 25/102 features complete (24.5%)
**Current**: 5/102 features complete (4.9%)

## Feature Selection Criteria

1. Already have CPU implementation with passing tests
2. Significant GPU acceleration potential (parallel operations)
3. Can share shader infrastructure patterns
4. High priority for typical OpenCV workflows

## Batch 1 Features (20)

### Filters (5 features)
1. ✅ **Box Blur** - CPU✓ Test✓ WASM✓ Gallery✓ | Need: GPU
2. ✅ **Median Blur** - CPU✓ WASM✓ Gallery✓ | Need: GPU, Test
3. ✅ **Bilateral Filter** - CPU✓ Test✓ WASM✓ Gallery✓ | Need: GPU
4. ✅ **Laplacian** - CPU✓ Test✓ WASM✓ Gallery✓ | Need: GPU
5. ✅ **Scharr** - CPU✓ WASM✓ Gallery✓ | Need: GPU, Test

### Geometric Transforms (3 features)
6. ✅ **Flip** - CPU✓ Test✓ WASM✓ Gallery✓ | Need: GPU
7. ✅ **Rotate** - CPU✓ Test✓ WASM✓ Gallery✓ | Need: GPU
8. ✅ **Warp Affine** - CPU✓ Test✓ WASM✓ Gallery✓ | Need: GPU

### Morphology (7 features)
9. ✅ **Erode** - CPU✓ Test✓ WASM✓ Gallery✓ | Need: GPU
10. ✅ **Dilate** - CPU✓ Test✓ WASM✓ Gallery✓ | Need: GPU
11. ✅ **Opening** - CPU✓ Test✓ WASM✓ Gallery✓ | Need: GPU
12. ✅ **Closing** - CPU✓ Test✓ WASM✓ Gallery✓ | Need: GPU
13. ✅ **Gradient** - CPU✓ Test✓ WASM✓ Gallery✓ | Need: GPU
14. ✅ **Top Hat** - CPU✓ WASM✓ Gallery✓ | Need: GPU, Test
15. ✅ **Black Hat** - CPU✓ WASM✓ Gallery✓ | Need: GPU, Test

### Threshold & Color (5 features)
16. ✅ **Adaptive Threshold** - CPU✓ WASM✓ Gallery✓ | Need: GPU, Test
17. ✅ **Histogram Equalization** - CPU✓ Test✓ WASM✓ Gallery✓ | Need: GPU
18. ✅ **RGB to Grayscale** - CPU✓ Test✓ WASM✓ Gallery✓ | Need: GPU
19. ✅ **RGB to HSV** - CPU✓ WASM✓ Gallery✓ | Need: GPU, Test
20. ✅ **RGB to Lab** - CPU✓ WASM✓ Gallery✓ | Need: GPU, Test

## Implementation Strategy

### Phase 1: GPU Shaders (Group by similarity)
- **Convolution filters**: Box, Bilateral, Laplacian, Scharr (share kernel application pattern)
- **Morphology**: Erode, Dilate, Opening, Closing, Gradient, Top/Black Hat (share structuring element pattern)
- **Geometric**: Flip, Rotate, Warp Affine (share texture sampling pattern)
- **Color**: RGB→Gray, RGB→HSV, RGB→Lab (share per-pixel conversion pattern)
- **Special**: Median (sorting network), Adaptive Threshold (local stats), Histogram Eq (global stats)

### Phase 2: GPU Integration
- Add async GPU functions to existing modules
- Update WASM bindings to use GPU-accelerated versions
- Add CPU/GPU selection logic with automatic fallback

### Phase 3: Testing
- Add missing unit tests (5 features need tests)
- Verify GPU output matches CPU output
- Benchmark GPU vs CPU performance (target >2x speedup)

### Phase 4: Verification
- Visual testing in web gallery
- Parameter variation testing
- Cross-browser compatibility

## Estimated Timeline

- Shaders: 2-3 hours (20 shaders, ~10min each)
- Integration: 1-2 hours (updating modules, exports)
- Testing: 1 hour (add missing tests, verify)
- Total: 4-6 hours for 20 features

## Success Criteria

Each feature must have:
- [x] CPU implementation
- [x] GPU WebGPU shader
- [x] WASM binding (GPU-enabled)
- [x] Unit tests passing
- [x] Gallery demo working
- [x] GPU speedup >2x (for applicable ops)

