#!/bin/bash
set -e

echo "Building opencv-rust for WASM with CPU multi-threading (no GPU)..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "wasm-pack not found. Installing..."
    cargo install wasm-pack
fi

# Build for web target with WASM and threading (CPU-only, no GPU)
echo "Compiling to WASM with rayon multi-threading..."
RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' \
wasm-pack build \
    --target web \
    --out-dir pkg \
    --features wasm-threading \
    --release \
    -- -Z build-std=panic_abort,std

echo ""
echo "✓ WASM build complete! Output in ./pkg/"
echo ""
echo "Features enabled:"
echo "  ✅ Rayon multi-threading (2-4x faster)"
echo "  ✅ All CPU operations parallelized"
echo "  ❌ GPU disabled (incompatible with threading)"
echo ""
echo "⚠️  IMPORTANT: Your web server must send these headers for threading to work:"
echo "  Cross-Origin-Opener-Policy: same-origin"
echo "  Cross-Origin-Embedder-Policy: require-corp"
echo ""
echo "To use in a web project:"
echo "  import init, { init_thread_pool, WasmMat, gaussian_blur_wasm } from './pkg/opencv_rust.js';"
echo "  await init();"
echo "  await init_thread_pool(navigator.hardwareConcurrency); // Initialize rayon threads"
echo ""
echo "To test the web demo:"
echo "  cd examples/web-benchmark"
echo "  bun install"
echo "  bun run dev"
