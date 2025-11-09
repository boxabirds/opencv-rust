#!/bin/bash
set -e

echo "Building opencv-rust for WASM..."

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
    --features wasm \
    --release

echo "✓ WASM build complete! Output in ./pkg/"
echo ""
echo "To use in a web project:"
echo "  import init, { WasmMat, gaussianBlur } from './pkg/opencv_rust.js';"
echo "  await init();"
echo "  await initGpu(); // Initialize WebGPU"
