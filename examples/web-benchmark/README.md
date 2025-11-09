# OpenCV-Rust Web Benchmark Demo

Interactive web-based benchmark for testing opencv-rust performance with WebGPU acceleration.

## Status

✅ **Fully Functional**: WASM bindings with WebGPU acceleration are implemented and working!

## Features

- 📤 Upload and process images in the browser ✅
- ⚡ Compare CPU (WASM) vs GPU (WebGPU) performance ✅
- 📊 Real-time performance metrics and results table ✅
- 🎨 Image preview ✅
- 🚀 Four operations: Gaussian Blur (GPU-accelerated), Resize, Threshold, Canny

## Requirements

### Browser Support

**Production Ready:**
- ✅ **Chrome/Edge 113+** (May 2023+) - WebGPU enabled by default, no flag needed
- ✅ **Chrome 121+** (Jan 2024+) - Recommended for best stability

**Experimental:**
- ⚠️ **Firefox Nightly** - Requires manual flag activation
- ⚠️ **Safari Technology Preview** - Limited/partial support

### Enable WebGPU (if needed)

**Modern Chrome/Edge (113+)**:
- WebGPU works out-of-the-box, no configuration needed!
- Just visit the demo and it should work

**Older Chrome/Edge (100-112)**:
1. Go to `chrome://flags/#enable-unsafe-webgpu`
2. Set to "Enabled"
3. Restart browser

**Firefox Nightly**:
1. Go to `about:config`
2. Set `dom.webgpu.enabled` to `true`
3. Restart browser

**Check WebGPU Support**:
Open browser console and run: `console.log('WebGPU:', !!navigator.gpu)`

## Installation

```bash
# Using Bun (recommended)
bun install

# Or using npm
npm install
```

## Development

```bash
# Start dev server with Bun
bun run dev

# Or with npm
npm run dev
```

Open http://localhost:3000 in your browser.

## Building for Production

```bash
# Build with Bun
bun run build

# Or with npm
npm run build

# Preview production build
bun run preview
```

## Usage

1. **Select an operation** (Gaussian Blur, Resize, Threshold, etc.)
2. **Upload an image** to process
3. **Run benchmark**:
   - CPU (WASM): Pure Rust compiled to WebAssembly
   - GPU (WebGPU): GPU-accelerated via WebGPU
   - Both: Run both and compare

4. **View results** in the table below

## Architecture

```
┌─────────────┐
│   React UI  │  User interaction & visualization
└──────┬──────┘
       │
┌──────▼──────┐
│ WASM Module │  opencv-rust compiled to WASM
└──────┬──────┘
       │
   ┌───┴────┐
   │        │
┌──▼───┐ ┌─▼────┐
│ CPU  │ │ GPU  │  Compute backends
│(WASM)│ │(WebGPU)│
└──────┘ └──────┘
```

## Building WASM Module

The WASM module with WebGPU support is already built! To rebuild:

```bash
# From opencv-rust root
./build-wasm-gpu.sh

# The pkg/ directory contains the compiled WASM module
# The web demo automatically references it via ../../../pkg/
```

## Performance Tips

- Use Chrome/Edge for best WebGPU support
- Test on different image sizes to see GPU scaling
- Larger images (1024x1024+) show GPU benefits
- Small images may be faster on CPU due to GPU overhead

## Troubleshooting

### WebGPU Not Available

**Check browser support**:
```javascript
console.log('WebGPU:', !!navigator.gpu);
```

**Common issues**:
- WebGPU flag not enabled
- Outdated GPU drivers
- Browser version too old
- Integrated GPU not supported

### WASM Not Loading

If you see errors loading the WASM module:
- Ensure `./build-wasm-gpu.sh` was run from the project root
- Check that `pkg/` directory exists with `opencv_rust_bg.wasm`
- Verify the import path in App.jsx points to `../../../pkg/opencv_rust.js`
- Run `npm run build` to verify the build succeeds

### Slow Performance

- Image too large (try < 2048x2048)
- Multiple operations running simultaneously
- Browser dev tools open (affects perf)
- Background processes consuming resources

## Comparison to Native Benchmarks

### Web (WASM + WebGPU)
- ✅ Runs in browser
- ✅ No installation needed
- ✅ Cross-platform
- ⚠️ ~2-3x slower than native
- ⚠️ Limited threading

### Native (Criterion)
- ✅ Full CPU/GPU performance
- ✅ Multi-threading with rayon
- ✅ More accurate measurements
- ⚠️ Requires Rust toolchain
- ⚠️ Platform-specific builds

## GPU Acceleration Status

**ALL Operations Fully GPU-Accelerated!** 🚀

- ✅ **Gaussian Blur** - GPU-accelerated with separable filter compute shaders
- ✅ **Resize** - GPU-accelerated with bilinear interpolation
- ✅ **Threshold** - GPU-accelerated binary thresholding
- ✅ **Canny Edge Detection** - GPU-accelerated with Sobel gradients + non-maximum suppression

All operations automatically use GPU when available, with seamless CPU fallback.

**Future Enhancements:**
- [ ] Create performance charts (Chart.js/D3)
- [ ] Add side-by-side before/after image comparison
- [ ] Support batch processing
- [ ] Export results to CSV/JSON
- [ ] Add more operations (HOG, SIFT, ORB, feature detection, etc.)
- [ ] Multi-pass edge tracking for full Canny algorithm

## Resources

- [WebGPU Spec](https://www.w3.org/TR/webgpu/)
- [wasm-pack Documentation](https://rustwasm.github.io/wasm-pack/)
- [wgpu for Web](https://wgpu.rs/)
- [React Documentation](https://react.dev/)

## License

Same as parent project (see root LICENSE)
