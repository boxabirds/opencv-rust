# OpenCV-Rust Web Benchmark

Interactive web demo showcasing OpenCV-Rust's GPU-accelerated image processing running in the browser via WebAssembly + WebGPU.

## Quick Start

```bash
npm install       # or: bun install
npm run dev      # or: bun run dev
# Open http://localhost:5173
```

## Smart WASM Rebuilding

`npm run dev` automatically checks if WASM needs rebuilding:

**✅ Fresh (skips rebuild):**
```
✅ WASM build is up to date, skipping rebuild
Server starts in ~1s
```

**🔨 Dirty (rebuilds):**
```
🔨 Source files newer than WASM, rebuild needed
Building WASM... (~12s)
```

**Triggers rebuild when:**
- WASM files missing
- `.rs` files modified
- `Cargo.toml/Cargo.lock` changed  
- Git working directory dirty (uncommitted Rust changes)

**Time saved per dev session:** 30-120 seconds

## Scripts

```bash
npm run dev              # Smart rebuild + dev server
npm run dev:check        # Check WASM freshness
npm run build:wasm       # Build WASM
npm run build:wasm:force # Force rebuild (bypass check)
npm run build            # Production build
```

## Features

- 50+ image processing operations
- GPU acceleration via WebGPU
- Real-time performance metrics
- CPU vs GPU comparison
- Operation history

See full documentation in main README.
