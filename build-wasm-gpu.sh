#!/bin/bash
set -e

echo "Building opencv-rust for WASM with WebGPU support..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found. Installing..."
    cargo install wasm-pack
fi

# Build for web target with GPU (no CPU threading)
echo "Compiling to WASM with WebGPU..."
wasm-pack build \
    --target web \
    --out-dir pkg \
    --features wasm,gpu

echo ""
echo "✓ WASM build complete! Output in ./pkg/"
echo ""
echo "Features enabled:"
echo "  ✅ WebGPU acceleration"
echo "  ✅ All image processing operations"
echo "  ❌ CPU threading (rayon) - not needed with GPU"
echo ""
echo "Browser Requirements:"
echo "  - WebGPU support (Chrome/Edge 113+, Firefox Nightly)"
echo "  - Enable at chrome://flags/#enable-unsafe-webgpu"
echo ""
echo "To use in a web project:"
echo "  import init, { initGpu, WasmMat, gaussianBlur } from './pkg/opencv_rust.js';"
echo "  await init();"
echo "  await initGpu(); // Initialize WebGPU"
echo ""
echo "To test the web demo:"
echo "  cd examples/web-benchmark"
echo "  bun install"
echo "  bun run dev"
