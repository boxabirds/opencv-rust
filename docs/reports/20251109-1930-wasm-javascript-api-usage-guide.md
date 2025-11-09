# WASM/JavaScript API Usage Guide

**Date**: 2025-11-09 19:30

## Overview

This guide explains how to use opencv-rust in the browser with WebAssembly and WebGPU acceleration.

## Important: Mat vs WasmMat

There are two Mat types when using opencv-rust in WASM:

### 1. WasmMat (JavaScript API)

**This is what you should use in JavaScript/TypeScript code.**

`WasmMat` is a JavaScript-friendly wrapper around the core `Mat` type, exposed through wasm-bindgen.

**Property Access (not functions!):**
```javascript
// ✅ CORRECT: Access width/height as properties
const width = mat.width;   // Getter property
const height = mat.height; // Getter property
const channels = mat.channels; // Getter property

// ❌ WRONG: Don't call as functions
const width = mat.width();  // ERROR: mat.width is not a function
const height = mat.height(); // ERROR: mat.height is not a function
```

**Creating a WasmMat:**
```javascript
import init, { WasmMat } from './opencv_rust.js';

// Initialize WASM module
await init();

// Create from dimensions
const mat = new WasmMat(640, 480, 3); // width, height, channels

// Create from ImageData
const imageData = canvas.getContext('2d').getImageData(0, 0, 640, 480);
const mat = WasmMat.fromImageData(
  imageData.data,
  imageData.width,
  imageData.height,
  4  // RGBA channels
);
```

**Converting back to ImageData:**
```javascript
// Get raw data from Mat
const data = mat.getData();

// Create ImageData
const imageData = new ImageData(
  new Uint8ClampedArray(data),
  mat.width,  // Property, not function
  mat.height  // Property, not function
);

// Draw to canvas
ctx.putImageData(imageData, 0, 0);
```

### 2. Mat (Rust API)

The core `Mat` struct has **methods** (functions):

```rust
// In Rust code
let mat = Mat::new_rows_cols(100, 200, 3, MatDepth::U8)?;
let width = mat.width();   // Function call
let height = mat.height(); // Function call
let cols = mat.cols();     // Function call
let rows = mat.rows();     // Function call
```

## WebGPU Initialization

**IMPORTANT: Initialize GPU before using GPU-accelerated operations:**

```javascript
import init, { initGpu } from './opencv_rust.js';

// 1. Initialize WASM module
await init();

// 2. Initialize WebGPU (async)
const gpuAvailable = await initGpu();

if (gpuAvailable) {
  console.log('✓ GPU acceleration enabled');
} else {
  console.log('⚠ Falling back to CPU');
}
```

### Common WebGPU Error

If you see this error:
```
Uncaught (in promise) Error: closure invoked recursively or after being dropped
```

**Cause**: The GPU is being initialized while WASM is still loading, or being called multiple times.

**Solution**:
```javascript
let gpuInitialized = false;

async function initializeOpenCV() {
  if (!gpuInitialized) {
    await init();
    await initGpu();
    gpuInitialized = true;
  }
}

// Call once at app startup
await initializeOpenCV();
```

## Image Processing Operations

All operations are **async** and return `Promise<WasmMat>`:

### Gaussian Blur

```javascript
import { gaussianBlur } from './opencv_rust.js';

// All operations are async and GPU-accelerated if available
const blurred = await gaussianBlur(mat, 5, 1.5);
// ksize=5, sigma=1.5

// Use the result
const data = blurred.getData();
const imageData = new ImageData(
  new Uint8ClampedArray(data),
  blurred.width,
  blurred.height
);
```

### Resize

```javascript
import { resize } from './opencv_rust.js';

const resized = await resize(mat, 320, 240);
// new width, new height
```

### Threshold

```javascript
import { threshold } from './opencv_rust.js';

const binary = await threshold(mat, 127, 255);
// threshold value, max value
```

### Canny Edge Detection

```javascript
import { canny } from './opencv_rust.js';

const edges = await canny(mat, 50, 150);
// low threshold, high threshold
```

## Complete Example

```javascript
import init, {
  WasmMat,
  initGpu,
  gaussianBlur,
  resize,
  threshold,
  canny,
  isGpuAvailable,
  getVersion
} from './opencv_rust.js';

async function processImage(imageElement) {
  // 1. Initialize (once per app)
  await init();
  await initGpu();

  console.log('OpenCV-Rust version:', getVersion());
  console.log('GPU available:', isGpuAvailable());

  // 2. Get image data from canvas
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = imageElement.width;
  canvas.height = imageElement.height;
  ctx.drawImage(imageElement, 0, 0);

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

  // 3. Create WasmMat from ImageData
  const mat = WasmMat.fromImageData(
    imageData.data,
    imageData.width,
    imageData.height,
    4  // RGBA
  );

  // 4. Process (all operations are async and GPU-accelerated)
  const blurred = await gaussianBlur(mat, 5, 1.5);
  const resized = await resize(blurred, 320, 240);
  const edges = await canny(resized, 50, 150);

  // 5. Convert back to ImageData
  function matToImageData(mat) {
    const data = mat.getData();
    return new ImageData(
      new Uint8ClampedArray(data),
      mat.width,   // Property, not function!
      mat.height   // Property, not function!
    );
  }

  const resultImageData = matToImageData(edges);

  // 6. Draw to canvas
  ctx.putImageData(resultImageData, 0, 0);

  // 7. Cleanup (optional, memory is auto-managed)
  mat.free();
  blurred.free();
  resized.free();
  edges.free();
}
```

## React/Vite Example

```javascript
import { useEffect, useState } from 'react';
import init, { WasmMat, initGpu, gaussianBlur } from './opencv_rust.js';

function App() {
  const [opencvReady, setOpencvReady] = useState(false);

  useEffect(() => {
    let initialized = false;

    async function initOpenCV() {
      if (!initialized) {
        await init();
        await initGpu();
        initialized = true;
        setOpencvReady(true);
      }
    }

    initOpenCV();
  }, []);

  async function processImage(imageData) {
    if (!opencvReady) {
      console.warn('OpenCV not ready yet');
      return;
    }

    const mat = WasmMat.fromImageData(
      imageData.data,
      imageData.width,
      imageData.height,
      4
    );

    const result = await gaussianBlur(mat, 5, 1.5);

    // ✅ CORRECT: Access as properties
    const width = result.width;
    const height = result.height;

    // ❌ WRONG: Don't call as functions
    // const width = result.width();  // ERROR!

    const data = result.getData();

    mat.free();
    result.free();

    return new ImageData(
      new Uint8ClampedArray(data),
      width,
      height
    );
  }

  return <div>...</div>;
}
```

## Common Mistakes

### ❌ Calling width/height as functions

```javascript
// WRONG
const width = mat.width();
const height = mat.height();
// Error: mat.width is not a function
```

**Fix:**
```javascript
// CORRECT
const width = mat.width;
const height = mat.height;
```

### ❌ Not awaiting async operations

```javascript
// WRONG
const blurred = gaussianBlur(mat, 5, 1.5);
const data = blurred.getData(); // Error: blurred is a Promise
```

**Fix:**
```javascript
// CORRECT
const blurred = await gaussianBlur(mat, 5, 1.5);
const data = blurred.getData();
```

### ❌ Initializing GPU multiple times

```javascript
// WRONG
await initGpu();
await initGpu(); // May cause "closure invoked recursively" error
```

**Fix:**
```javascript
// CORRECT
let gpuInitialized = false;
if (!gpuInitialized) {
  await initGpu();
  gpuInitialized = true;
}
```

### ❌ Using Mat instead of WasmMat in JavaScript

```javascript
// WRONG - Mat is not exposed to JavaScript
import { Mat } from './opencv_rust.js'; // Doesn't exist
```

**Fix:**
```javascript
// CORRECT
import { WasmMat } from './opencv_rust.js';
```

## TypeScript Definitions

If using TypeScript, add these definitions:

```typescript
declare module './opencv_rust.js' {
  export default function init(): Promise<void>;

  export class WasmMat {
    constructor(width: number, height: number, channels: number);
    static fromImageData(
      data: Uint8Array,
      width: number,
      height: number,
      channels: number
    ): WasmMat;

    getData(): Uint8Array;

    // Properties (not methods!)
    readonly width: number;
    readonly height: number;
    readonly channels: number;

    free(): void;
  }

  export function initGpu(): Promise<boolean>;
  export function isGpuAvailable(): boolean;
  export function getVersion(): string;

  export function gaussianBlur(
    src: WasmMat,
    ksize: number,
    sigma: number
  ): Promise<WasmMat>;

  export function resize(
    src: WasmMat,
    width: number,
    height: number
  ): Promise<WasmMat>;

  export function threshold(
    src: WasmMat,
    thresh: number,
    maxVal: number
  ): Promise<WasmMat>;

  export function canny(
    src: WasmMat,
    threshold1: number,
    threshold2: number
  ): Promise<WasmMat>;
}
```

## Performance Tips

1. **Enable GPU acceleration**: Always call `initGpu()` for 10-100x speedup
2. **Reuse Mat objects**: Avoid creating/destroying Mats in tight loops
3. **Batch operations**: Process multiple images in parallel using `Promise.all()`
4. **Free memory**: Call `.free()` when done to release WASM memory immediately
5. **Use Web Workers**: Offload processing to workers for better responsiveness

## Debugging

Enable console logging:

```javascript
// All operations log to console
await gaussianBlur(mat, 5, 1.5);
// Console: "✓ GPU blur completed in 2.3ms" (or falls back to CPU)
```

Check GPU status:

```javascript
import { isGpuAvailable } from './opencv_rust.js';

if (isGpuAvailable()) {
  console.log('✓ Using GPU acceleration');
} else {
  console.log('⚠ Using CPU (slower)');
}
```

## Browser Compatibility

- **WebGPU**: Chrome 113+, Edge 113+, Safari 18+ (macOS 15+)
- **WebAssembly**: All modern browsers
- **Fallback**: Automatically uses CPU if GPU unavailable

## Summary

**Key Points:**
- ✅ Use `WasmMat` in JavaScript (not `Mat`)
- ✅ Access `width`, `height`, `channels` as **properties** (not functions)
- ✅ All operations are **async** (use `await`)
- ✅ Initialize GPU once with `initGpu()` for best performance
- ✅ Call `.free()` to release memory when done
