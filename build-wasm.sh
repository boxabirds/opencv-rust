#!/bin/bash
set -e

echo "Building opencv-rust for WASM with threading support..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found. Installing..."
    cargo install wasm-pack
fi

# Build for web target with WASM and GPU features
echo "Compiling to WASM with WebGPU support..."
wasm-pack build \
    --target web \
    --out-dir pkg \
    --features wasm

echo ""
echo "✓ WASM build complete! Output in ./pkg/"
echo ""
echo "⚠️  IMPORTANT: Your web server must send these headers for threading to work:"
echo "  Cross-Origin-Opener-Policy: same-origin"
echo "  Cross-Origin-Embedder-Policy: require-corp"
echo ""
echo "To use in a web project:"
echo "  import init, { initThreadPool, initGpu, WasmMat, gaussianBlur } from './pkg/opencv_rust.js';"
echo "  await init();"
echo "  await initThreadPool(navigator.hardwareConcurrency); // Initialize rayon threads"
echo "  await initGpu(); // Initialize WebGPU (optional)"
