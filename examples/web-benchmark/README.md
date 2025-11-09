# OpenCV-Rust Web Benchmark Demo

Interactive web-based benchmark for testing opencv-rust performance with WebGPU acceleration.

## Status

🚧 **Work in Progress**: This demo is a placeholder structure. WASM bindings are not yet implemented.

## Features (Planned)

- 📤 Upload and process images in the browser
- ⚡ Compare CPU (WASM) vs GPU (WebGPU) performance
- 📊 Real-time performance metrics and charts
- 🎨 Visual before/after comparison
- 📥 Export benchmark results

## Requirements

### Browser Support

- **Chrome/Edge 113+** with WebGPU flag enabled
- **Firefox Nightly** with WebGPU behind flag (experimental)
- **Safari Technology Preview** (limited support)

### Enable WebGPU

**Chrome/Edge**:
1. Go to `chrome://flags/#enable-unsafe-webgpu`
2. Set to "Enabled"
3. Restart browser

**Firefox**:
1. Go to `about:config`
2. Set `dom.webgpu.enabled` to `true`
3. Restart browser

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

## Building WASM Module (Future)

When WASM support is complete, build with:

```bash
# From opencv-rust root
wasm-pack build --features wasm --target web

# Link to web demo
cd examples/web-benchmark
ln -s ../../pkg opencv-rust-wasm
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

**Future**: When WASM is implemented, check:
- WASM module built correctly
- CORS headers configured
- Module path correct in package.json

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

## Next Steps

- [ ] Implement WASM bindings for opencv-rust
- [ ] Add WebGPU compute shader compilation
- [ ] Create performance charts (Chart.js/D3)
- [ ] Add side-by-side image comparison
- [ ] Support batch processing
- [ ] Export results to CSV/JSON
- [ ] Add more operations (Canny, Resize, Threshold)

## Resources

- [WebGPU Spec](https://www.w3.org/TR/webgpu/)
- [wasm-pack Documentation](https://rustwasm.github.io/wasm-pack/)
- [wgpu for Web](https://wgpu.rs/)
- [React Documentation](https://react.dev/)

## License

Same as parent project (see root LICENSE)
