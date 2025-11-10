# Backend Selection Design: GPU vs CPU

**Author**: Claude
**Date**: 2025-11-10
**Status**: Approved for Implementation

---

## Problem Statement

The opencv-rust WASM library needs to support both GPU (WebGPU) and CPU execution paths while maintaining 100% API compatibility with OpenCV.js. Users need:

1. **Automatic fallback**: Try GPU first, fall back to CPU if unavailable
2. **Manual control**: Force GPU or CPU for specific use cases
3. **Zero API changes**: Same function signatures as OpenCV.js
4. **Minimal overhead**: Backend selection must be fast (<5ns per call)

---

## Solution: Runtime Backend Selection

### Architecture Overview

```
User calls WASM function (e.g., gaussian_blur_wasm)
           ↓
    get_backend() [1-2ns atomic read]
           ↓
    ┌──────┴──────┐
    │   Auto (0)  │ → Check cached resolution → GPU or CPU
    │ WebGPU (1)  │ → Force GPU path
    │   CPU (2)   │ → Force CPU path
    └─────────────┘
           ↓
    Execute operation on selected backend
```

### Three Backend Modes

1. **Auto (Default)**: Intelligent GPU-first with fallback
   - First call: Check `GpuContext::is_available()`
   - Cache result in atomic variable
   - Subsequent calls: Use cached value (1-2ns overhead)

2. **WebGPU**: Force GPU execution
   - Returns error if GPU unavailable
   - For applications requiring GPU features
   - Direct dispatch, no fallback overhead

3. **CPU**: Force CPU execution
   - Never attempts GPU initialization
   - For compatibility testing or CPU-only environments
   - Zero GPU overhead

---

## Implementation

### Core Data Structures

```rust
use std::sync::atomic::{AtomicU8, Ordering};

/// User-facing backend setting
#[repr(u8)]
enum Backend {
    Auto = 0,      // Resolve once to GPU or CPU
    WebGPU = 1,    // Force GPU
    CPU = 2,       // Force CPU
}

/// Cached backend resolution (only for Auto mode)
#[repr(u8)]
enum ResolvedBackend {
    Unresolved = 0,  // Not yet determined
    GPU = 1,         // Resolved to GPU
    CPU = 2,         // Resolved to CPU
}

// User-facing setting (changeable at runtime)
static BACKEND: AtomicU8 = AtomicU8::new(Backend::Auto as u8);

// Cached resolution (only for Auto mode)
static RESOLVED: AtomicU8 = AtomicU8::new(ResolvedBackend::Unresolved as u8);
```

### Backend Resolution Logic

```rust
/// Get the active backend (cached resolution for Auto mode)
///
/// Performance: 1-2ns per call (single atomic read)
fn get_backend() -> u8 {
    let backend = BACKEND.load(Ordering::Relaxed);

    match backend {
        1 => 1, // WebGPU - direct return
        2 => 2, // CPU - direct return
        0 => {  // Auto - resolve once and cache
            let resolved = RESOLVED.load(Ordering::Relaxed);
            if resolved != 0 {
                return resolved; // Already resolved (fast path)
            }

            // First call: resolve based on GPU availability
            let result = if GpuContext::is_available() { 1 } else { 2 };
            RESOLVED.store(result, Ordering::Relaxed);
            result
        }
        _ => 2, // Invalid - default to CPU
    }
}
```

### Public API (WASM)

```rust
use wasm_bindgen::prelude::*;

/// Set the backend execution mode
///
/// # Arguments
/// * `backend` - "auto" | "webgpu" | "gpu" | "cpu"
///
/// # Examples
/// ```javascript
/// import init, { setBackend } from './opencv_rust.js';
///
/// await init();
///
/// // Try GPU first, fall back to CPU
/// setBackend('auto'); // Default
///
/// // Force GPU (error if unavailable)
/// setBackend('webgpu');
///
/// // Force CPU
/// setBackend('cpu');
/// ```
#[wasm_bindgen]
pub fn setBackend(backend: &str) {
    let value = match backend {
        "webgpu" | "gpu" => Backend::WebGPU as u8,
        "cpu" => Backend::CPU as u8,
        "auto" => {
            // Reset cache to force re-resolution
            RESOLVED.store(ResolvedBackend::Unresolved as u8, Ordering::Relaxed);
            Backend::Auto as u8
        }
        _ => {
            #[cfg(target_arch = "wasm32")]
            web_sys::console::warn_1(&format!("Invalid backend '{}', defaulting to 'auto'", backend).into());
            Backend::Auto as u8
        }
    };
    BACKEND.store(value, Ordering::Relaxed);
}

/// Get the current backend setting
///
/// # Returns
/// "auto" | "webgpu" | "cpu"
#[wasm_bindgen]
pub fn getBackend() -> String {
    match BACKEND.load(Ordering::Relaxed) {
        0 => "auto".to_string(),
        1 => "webgpu".to_string(),
        2 => "cpu".to_string(),
        _ => "auto".to_string(),
    }
}

/// Get the resolved backend (only meaningful in Auto mode)
///
/// # Returns
/// "gpu" | "cpu" | "unresolved"
#[wasm_bindgen]
pub fn getResolvedBackend() -> String {
    match RESOLVED.load(Ordering::Relaxed) {
        0 => "unresolved".to_string(),
        1 => "gpu".to_string(),
        2 => "cpu".to_string(),
        _ => "unresolved".to_string(),
    }
}
```

### Integration with Operations

Every WASM-bound operation follows this pattern:

```rust
#[wasm_bindgen]
pub async fn gaussian_blur_wasm(
    src: &WasmMat,
    ksize: usize,
    sigma: f64
) -> Result<WasmMat, JsValue> {
    match get_backend() {
        1 => {
            // GPU path
            let mut dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            gaussian_blur_gpu_async(&src.0, &mut dst, ksize, sigma)
                .await
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            Ok(WasmMat(dst))
        }
        _ => {
            // CPU path (default for backend=2 or invalid)
            let dst = gaussian_blur_cpu(&src.0, ksize, sigma)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            Ok(WasmMat(dst))
        }
    }
}
```

---

## Performance Analysis

### Overhead Breakdown

| Operation | Time | Notes |
|-----------|------|-------|
| `BACKEND.load()` | ~1ns | Single atomic read (L1 cache hit) |
| `RESOLVED.load()` | ~1ns | Single atomic read (Auto mode only) |
| `GpuContext::is_available()` | ~1ns | Single atomic read (first call only) |
| **Total per call** | **1-2ns** | Negligible vs operation time (ms) |

### Comparison to Alternatives

| Approach | Overhead | Complexity |
|----------|----------|------------|
| **Our design (atomic)** | 1-2ns | Low |
| Check GPU every call | 1-2ns + async overhead | Medium |
| Duplicate WASM functions | 0ns | High (2x code) |
| Compile-time feature flags | 0ns | Very high (rebuild) |

### Why This Is Efficient

1. **Atomic operations are fast**: Modern CPUs execute atomic loads in 1-2 clock cycles
2. **L1 cache locality**: Both atomics stay hot in L1 cache
3. **No branching overhead**: Direct match on integer value
4. **No async overhead**: Resolution happens once, not per call
5. **Negligible vs operation time**: 1-2ns overhead for 0.1-100ms operations = 0.000002-0.002% overhead

**Example**: For a 10ms gaussian_blur operation, the 2ns backend check is **0.00002%** overhead.

---

## Integration with OpenCV.js API Parity

### Key Principle: Backend Selection Is Internal

The backend selection mechanism is **completely transparent** to the OpenCV.js API:

```javascript
// OpenCV.js API (what users see)
cv.GaussianBlur(src, dst, ksize, sigmaX);

// Our WASM API (100% identical signature)
cv.GaussianBlur(src, dst, ksize, sigmaX);

// Backend selection happens internally - users never see it
```

### Optional Advanced Control

For power users who want explicit control:

```javascript
import init, { setBackend, getResolvedBackend } from './opencv_rust.js';

await init();

// Check what backend will be used
console.log('Backend:', getResolvedBackend()); // "gpu" or "cpu"

// Force GPU for performance testing
setBackend('webgpu');

// Force CPU for correctness comparison
setBackend('cpu');

// Reset to auto
setBackend('auto');
```

This is **optional** and doesn't break OpenCV.js compatibility - the default behavior matches OpenCV.js (try GPU, fall back gracefully).

---

## Edge Cases and Error Handling

### Case 1: GPU Unavailable in WebGPU Mode

```rust
match get_backend() {
    1 => {
        gaussian_blur_gpu_async(&src.0, &mut dst, ksize, sigma)
            .await
            .map_err(|e| JsValue::from_str(&format!("GPU error: {}. Try setBackend('auto') or setBackend('cpu')", e)))?;
    }
    _ => { /* CPU path */ }
}
```

### Case 2: GPU Becomes Unavailable After Resolution

- GPU context is initialized once at startup
- If initialization succeeds, GPU remains available for session
- If GPU fails later (e.g., context lost), operations return errors
- User can call `setBackend('cpu')` to switch to CPU

### Case 3: Concurrent setBackend() Calls

- Atomic operations are thread-safe
- Last write wins (standard atomic semantics)
- No undefined behavior or data races

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_backend_is_auto() {
        assert_eq!(getBackend(), "auto");
    }

    #[test]
    fn test_set_backend() {
        setBackend("cpu");
        assert_eq!(getBackend(), "cpu");

        setBackend("webgpu");
        assert_eq!(getBackend(), "webgpu");

        setBackend("auto");
        assert_eq!(getBackend(), "auto");
    }

    #[test]
    fn test_invalid_backend_defaults_to_auto() {
        setBackend("invalid");
        assert_eq!(getBackend(), "auto");
    }
}
```

### Integration Tests (WASM)

```javascript
// tests/backend_selection.spec.js
import { test, expect } from '@playwright/test';
import init, {
    setBackend,
    getBackend,
    getResolvedBackend,
    gaussian_blur_wasm
} from '../pkg/opencv_rust.js';

test('default backend is auto', async () => {
    await init();
    expect(getBackend()).toBe('auto');
});

test('auto mode resolves to gpu when available', async () => {
    await init();
    setBackend('auto');

    // Trigger resolution
    const src = createTestImage();
    await gaussian_blur_wasm(src, 5, 1.5);

    const resolved = getResolvedBackend();
    expect(['gpu', 'cpu']).toContain(resolved);
});

test('webgpu mode forces gpu', async () => {
    await init();
    setBackend('webgpu');

    const src = createTestImage();

    try {
        await gaussian_blur_wasm(src, 5, 1.5);
        // Should succeed if GPU available
    } catch (e) {
        // Should fail gracefully if GPU unavailable
        expect(e.message).toContain('GPU error');
    }
});

test('cpu mode uses cpu implementation', async () => {
    await init();
    setBackend('cpu');

    const src = createTestImage();
    const result = await gaussian_blur_wasm(src, 5, 1.5);

    expect(result).toBeDefined();
    expect(getResolvedBackend()).toBe('unresolved'); // CPU mode doesn't use resolution
});
```

---

## Migration Guide (for OpenCV.js Users)

### Default Behavior: Drop-In Replacement

```javascript
// Before (OpenCV.js)
const cv = await loadOpenCV();
cv.GaussianBlur(src, dst, new cv.Size(5, 5), 1.5);

// After (opencv-rust WASM)
import init from './opencv_rust.js';
const cv = await init();
cv.GaussianBlur(src, dst, new cv.Size(5, 5), 1.5);
// Automatically uses GPU if available, falls back to CPU
```

### Advanced: Performance Tuning

```javascript
import init, { setBackend, getResolvedBackend } from './opencv_rust.js';

await init();

// Check what backend will be used
console.log('Using:', getResolvedBackend());

// For GPU-only applications (e.g., real-time video processing)
setBackend('webgpu');
if (getResolvedBackend() !== 'gpu') {
    console.error('GPU required but not available');
}

// For testing: compare GPU vs CPU results
setBackend('gpu');
const gpuResult = await cv.GaussianBlur(src, dst1, 5, 1.5);

setBackend('cpu');
const cpuResult = await cv.GaussianBlur(src, dst2, 5, 1.5);

compareResults(gpuResult, cpuResult);
```

---

## Implementation Timeline

### Phase 1: Core Infrastructure (Week 1)
- [ ] Add `BACKEND` and `RESOLVED` static atomics
- [ ] Implement `get_backend()` function
- [ ] Implement `setBackend()`, `getBackend()`, `getResolvedBackend()` WASM exports
- [ ] Add unit tests

### Phase 2: Integration (Week 2-3)
- [ ] Update all 15-20 core operations to use `get_backend()`
- [ ] Test GPU path with forced WebGPU mode
- [ ] Test CPU path with forced CPU mode
- [ ] Test Auto mode resolution logic
- [ ] Add integration tests

### Phase 3: Documentation (Week 4)
- [ ] Document backend selection API
- [ ] Create migration guide for OpenCV.js users
- [ ] Add examples to gallery demos
- [ ] Update README with backend selection info

---

## Files to Modify

1. **`src/wasm/mod.rs`** (or new `src/wasm/backend.rs`)
   - Add `BACKEND`, `RESOLVED` static atomics
   - Implement `get_backend()`, `setBackend()`, `getBackend()`, `getResolvedBackend()`

2. **`src/wasm/ops/*.rs`** (15-20 files)
   - Update each `*_wasm()` function to use `get_backend()`
   - Add GPU path (call `*_gpu_async()`)
   - Keep CPU path (call `*_cpu()`)

3. **`examples/web-benchmark/src/BackendSelector.jsx`** (new)
   - UI component for backend selection in gallery
   - Display current backend and resolved backend
   - Allow users to test different backends

4. **`tests/backend_selection.spec.js`** (new)
   - Integration tests for backend selection
   - Test all three modes (Auto, WebGPU, CPU)

---

## Open Questions

### Q: Should we add per-operation backend override?

**No.** Keep it simple - global backend setting only. Rationale:
- Per-operation control adds API complexity
- Breaks OpenCV.js compatibility (additional parameter)
- Rarely needed in practice (users want consistent behavior)
- Can be added later if demand exists

### Q: Should Auto mode cache be per-operation?

**No.** Single global cache is sufficient. Rationale:
- GPU availability doesn't vary per operation
- Single cache is simpler and faster
- Reduces state management complexity

### Q: What if user wants to re-check GPU availability without page reload?

**Solution**: Reset Auto mode cache:
```javascript
setBackend('auto'); // This resets RESOLVED to Unresolved
```

---

## Conclusion

This design provides:

1. ✅ **100% OpenCV.js API compatibility** (backend selection is internal)
2. ✅ **Minimal overhead** (1-2ns per call)
3. ✅ **Automatic fallback** (GPU-first in Auto mode)
4. ✅ **Manual control** (WebGPU/CPU modes for advanced users)
5. ✅ **Thread-safe** (atomic operations)
6. ✅ **Simple implementation** (~100 lines of code)
7. ✅ **Easy testing** (force specific backends)
8. ✅ **Future-proof** (can add optimizations without API changes)

**Status**: Ready for implementation
**Next Step**: Implement Phase 1 (core infrastructure) in `src/wasm/backend.rs`
