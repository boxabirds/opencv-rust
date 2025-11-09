# Benchmarking Guide

This document explains how to benchmark opencv-rust performance for both CPU and GPU implementations.

## Native Benchmarks (Using Criterion)

### CPU-Only Benchmarks

Run all CPU benchmarks with rayon parallelization:

```bash
cargo bench
```

This will benchmark all 20 operations and generate HTML reports in `target/criterion/`.

### GPU Benchmarks

Run benchmarks with GPU acceleration enabled:

```bash
cargo bench --features gpu
```

This will include additional GPU-specific benchmarks:
- **Gaussian Blur (GPU)**: Tests GPU performance on 512x512, 1024x1024, and 2048x2048 images
- **GPU vs CPU Comparison**: Direct comparison of GPU-accelerated vs rayon-optimized CPU

**Note**: GPU benchmarks require compatible GPU hardware:
- **Linux**: Vulkan-capable GPU
- **macOS**: Metal-capable GPU (M1/M2 or AMD/NVIDIA)
- **Windows**: DirectX 12 capable GPU

If no GPU is available, the GPU benchmarks will be skipped gracefully.

### Viewing Results

After running benchmarks, open the generated HTML reports:

```bash
# Open the main index
open target/criterion/report/index.html

# Or view specific operation results
open target/criterion/Gaussian\ Blur\ \(GPU\)/report/index.html
```

### Running Specific Benchmarks

```bash
# Only Gaussian blur benchmarks
cargo bench --features gpu gaussian_blur

# Only GPU benchmarks
cargo bench --features gpu "Gaussian Blur (GPU)"

# Only adaptive threshold
cargo bench adaptive_threshold
```

## Web Benchmarks (WASM + WebGPU)

**Current Status**: Native benchmarks only. Web benchmarking requires additional setup.

To run benchmarks in the browser, you would need:

1. **Compile to WASM with WebGPU**:
   ```bash
   # Future capability - not yet implemented
   wasm-pack build --features wasm --target web
   ```

2. **Create web demo** with custom benchmark UI:
   - Use `performance.now()` for timing
   - Display results in interactive charts
   - Compare CPU (WASM) vs GPU (WebGPU) performance

3. **Browser Requirements**:
   - **Chrome/Edge**: WebGPU enabled (chrome://flags/#enable-unsafe-webgpu)
   - **Firefox**: WebGPU behind flag
   - **Safari**: Limited WebGPU support (experimental)

### Web Demo (Coming Soon)

A Bun + React web demo is planned that will allow you to:
- Upload test images
- Run operations with CPU and GPU
- See real-time performance comparisons
- View before/after results
- Export benchmark data

**Tracked in**: Phase 6 of GPU implementation plan

## Performance Expectations

### CPU Performance (with rayon)

Current performance vs C++ OpenCV (as of 2025-01-09):
- **Average**: 1.5x faster than C++ OpenCV
- **Best**: Canny edge detection (3.3x faster)
- **Matched**: Most operations within 10% of C++ performance

See `docs/reports/20251109-0918-rayon-optimization-results.md` for details.

### GPU Performance (WebGPU)

**Expected** (not yet measured on real hardware):
- **10-15x speedup** for Gaussian blur on mid-range GPU
- **50-100x speedup** for large images (2048x2048+) on high-end GPU
- GPU overhead makes small images (< 256x256) slower than CPU

## Benchmarking Tips

### For Accurate Results

1. **Close unnecessary applications** to reduce system noise
2. **Disable power-saving modes** (use high-performance mode)
3. **Run multiple times** for consistency:
   ```bash
   cargo bench --features gpu -- --sample-size 100
   ```

4. **Warm up the GPU** before measuring (Criterion does this automatically)

### Interpreting Results

- **Time**: Lower is better (milliseconds per iteration)
- **Throughput**: Higher is better (iterations per second)
- **R²**: Closer to 1.0 means more reliable measurements
- **Mean vs Median**: Use median for skewed distributions

### Comparing CPU vs GPU

GPU will show benefits when:
- **Large images**: 1024x1024 or larger
- **Large kernels**: 7x7 Gaussian blur or larger
- **Batch processing**: Multiple images sequentially

CPU (rayon) may be faster when:
- **Small images**: < 512x512
- **Small kernels**: 3x3 filters
- **Single operation**: GPU initialization overhead

## Troubleshooting

### GPU Benchmarks Not Running

**Symptom**: "GPU not available - skipping GPU benchmarks"

**Solutions**:
1. Check GPU drivers are installed:
   ```bash
   # Linux
   vulkaninfo

   # macOS (Metal is built-in)
   system_profiler SPDisplaysDataType

   # Windows
   dxdiag
   ```

2. Verify wgpu can detect GPU:
   ```bash
   cargo run --features gpu --example init_gpu
   ```

3. Try forcing a specific backend:
   ```bash
   WGPU_BACKEND=vulkan cargo bench --features gpu
   # or: WGPU_BACKEND=metal (macOS)
   # or: WGPU_BACKEND=dx12 (Windows)
   ```

### Benchmarks Taking Too Long

Reduce sample size:
```bash
cargo bench --features gpu -- --sample-size 10
```

Or benchmark specific operations only:
```bash
cargo bench --features gpu gaussian_blur
```

### Out of Memory Errors

Reduce test image sizes or run benchmarks sequentially:
```bash
cargo bench --features gpu -- --test-threads=1
```

## Next Steps

- [ ] Implement web-based benchmarking with Bun + React
- [ ] Add more GPU operations (resize, threshold, morphology)
- [ ] Create comparison charts (GPU vs CPU vs C++ OpenCV)
- [ ] Add memory usage profiling
- [ ] Benchmark WASM performance in browser
