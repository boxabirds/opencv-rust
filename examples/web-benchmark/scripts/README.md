# Web Benchmark Scripts

## check-wasm.js

Checks if the WASM build is up to date before running dev server.

### What it checks:

1. **WASM output exists** - `../../pkg/opencv_rust_bg.wasm` must exist
2. **Git status** - No uncommitted changes to Rust files (`src/`, `Cargo.toml`, `Cargo.lock`)
3. **Source timestamps** - All `.rs` files must be older than WASM output
4. **Cargo files** - `Cargo.toml` and `Cargo.lock` must be older than WASM output

### Exit codes:

- `0` - WASM is fresh, no rebuild needed
- `1` - WASM is dirty, rebuild needed

### Usage:

The script is automatically run by `npm run dev`:

```bash
# Checks WASM freshness, rebuilds only if needed
npm run dev

# Force rebuild regardless of freshness
npm run build:wasm:force
```

### Manual usage:

```bash
# Check freshness
node scripts/check-wasm.js && echo "Fresh" || echo "Dirty"

# Check and rebuild if needed
node scripts/check-wasm.js || npm run build:wasm
```

### Why this matters:

WASM builds take 12-30 seconds. By checking freshness:
- **First run**: Rebuilds (needed)
- **Subsequent runs**: Skips rebuild (~instant startup)
- **After code changes**: Detects and rebuilds
- **After git pull**: Detects and rebuilds

**Time saved per dev session:** 30-120 seconds
