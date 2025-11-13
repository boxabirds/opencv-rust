# WASM Threading Issue

## Problem
Gaussian blur tests hang indefinitely when running in WASM.

## Root Cause
The codebase uses `rayon` for parallelization throughout (filter.rs, edge.rs, color.rs, etc.). These files use `par_chunks_mut()` and other rayon APIs unconditionally without `#[cfg(feature = "rayon")]` guards.

When building for WASM:
1. If rayon is included (default feature): Code compiles but hangs at runtime because `par_chunks_mut()` waits for thread pool initialization that never happens
2. If rayon is excluded: Code doesn't compile due to missing rayon APIs

## What Was Tried
- Building without rayon: 27 compile errors
- Building with rayon but no threading: Hangs waiting for thread pool
- Building with wasm-threading: Requires CORS headers (Cross-Origin-Opener-Policy + Cross-Origin-Embedder-Policy) that Playwright's simple Python server doesn't provide

## Solution Required
Add conditional compilation throughout the codebase:
```rust
#[cfg(feature = "rayon")]
use rayon::prelude::*;

// Then in functions:
#[cfg(feature = "rayon")]
{
    data.par_chunks_mut(size).for_each(|chunk| { ... });
}
#[cfg(not(feature = "rayon"))]
{
    data.chunks_mut(size).for_each(|chunk| { ... });
}
```

This needs to be applied to:
- src/imgproc/filter.rs
- src/imgproc/edge.rs
- src/imgproc/color.rs
- src/imgproc/geometric.rs
- src/imgproc/threshold.rs
- src/features2d/keypoints.rs
- src/imgproc/advanced_filter.rs

## Test Status
All Gaussian blur parity tests are currently blocked on this issue.
