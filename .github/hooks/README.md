# Git Hooks

This directory contains Git hooks that help maintain code quality.

## Installation

### Automatic (Recommended)

Run this from the repository root:

```bash
./scripts/install-hooks.sh
```

### Manual

```bash
# Pre-push hook (validates WASM builds)
ln -s ../../.github/hooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

## Available Hooks

### pre-push

Runs before `git push` to validate WASM builds.

**What it does:**
- Checks if WASM-related files were modified
- Runs `wasm-pack build` to verify the WASM module compiles
- Verifies output artifacts exist

**Requirements:**
- `wasm-pack` installed (optional - hook skips if not found)
- Rust toolchain with `wasm32-unknown-unknown` target

**Bypass:**
```bash
git push --no-verify  # Use in emergencies only
```

## Why Use Hooks?

Git hooks catch issues **before** they reach CI:
- ⚡ Faster feedback (local vs waiting for CI)
- 💰 Saves CI minutes
- 🚫 Prevents broken code from being pushed
- 🎯 Catches platform-specific issues early

## Troubleshooting

### Hook not running

```bash
# Check if hook is installed
ls -la .git/hooks/pre-push

# Reinstall if needed
ln -sf ../../.github/hooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

### Hook taking too long

The pre-push hook only runs if you've modified WASM-related files. If it's slow:
- Enable caching: hooks use incremental compilation
- Skip hook: `git push --no-verify` (not recommended)
- Disable hook: `rm .git/hooks/pre-push`
