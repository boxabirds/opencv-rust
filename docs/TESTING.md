# Testing Guide

This document describes the testing infrastructure for opencv-rust.

## Overview

opencv-rust has multiple test suites to ensure code quality across different platforms and backends:

1. **Native Rust Tests** - Standard Rust unit and integration tests
2. **WASM Tests** - Browser-based tests using `wasm-bindgen-test`
3. **Accuracy Tests** - Compare outputs against reference implementations
4. **GPU Tests** - Validate WebGPU backend functionality

## Running Tests Locally

### Native Tests

```bash
# Run all native tests
cargo test

# Run specific test
cargo test test_gaussian_blur

# Run with GPU features
cargo test --features gpu
```

### WASM Tests

```bash
# Install wasm-pack (one-time setup)
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Add WASM target
rustup target add wasm32-unknown-unknown

# Run WASM tests in Firefox (headless)
wasm-pack test --headless --firefox --features wasm

# Run WASM tests in Chrome (headless)
wasm-pack test --headless --chrome --features wasm

# Run specific WASM test
wasm-pack test --headless --firefox --features wasm -- --test wasm_gaussian_blur_tests
```

### Running Tests in Browser (Interactive)

```bash
# Firefox
wasm-pack test --firefox --features wasm

# Chrome
wasm-pack test --chrome --features wasm
```

This opens a browser window where you can see test output in the console and use browser DevTools for debugging.

## Continuous Integration

### GitHub Actions

All tests run automatically on push and pull requests via GitHub Actions:

- **WASM Build Check** (`.github/workflows/wasm-build.yml`)
  - Verifies WASM module compiles successfully
  - Runs on: every push and PR
  - Platforms: Ubuntu (fastest)

- **WASM Tests** (`.github/workflows/wasm-tests.yml`)
  - Runs full WASM test suite in headless browsers
  - Runs on: every push and PR
  - Browsers: Firefox and Chrome

### Viewing CI Results

1. Go to the [Actions tab](https://github.com/boxabirds/opencv-rust/actions)
2. Click on a workflow run
3. Expand the failed step to see error details

## Git Hooks (Local Validation)

Git hooks provide fast feedback before pushing code:

### Install Hooks

```bash
# Automatic installation
./scripts/install-hooks.sh

# Manual installation
ln -s ../../.github/hooks/pre-push .git/hooks/pre-push
chmod +x .git/hooks/pre-push
```

### Pre-Push Hook

Runs automatically before `git push`:
- Validates WASM builds if WASM-related files changed
- Catches build errors before they reach CI
- Can be bypassed with `git push --no-verify` (not recommended)

## Test Organization

### Directory Structure

```
tests/
├── wasm_*_tests.rs          # WASM-specific tests
├── test_accuracy_*.rs        # Accuracy validation tests
├── test_*.rs                 # Native integration tests
└── wasm_test_utils/          # Shared utilities for WASM tests
```

### Test Naming Conventions

- **Native tests**: `test_<operation>_<scenario>`
- **WASM tests**: `test_<operation>_<scenario>` in `wasm_*_tests.rs`
- **Accuracy tests**: `test_accuracy_<operation>_<scenario>`

## Writing Tests

### WASM Test Template

Use `docs/WASM_TEST_TEMPLATE.rs` as a starting point for new WASM tests:

```bash
cp docs/WASM_TEST_TEMPLATE.rs tests/wasm_my_operation_tests.rs
# Edit the template, replacing placeholders
```

### Test Categories

Each operation should have tests covering:

1. **Smoke tests** - Basic functionality doesn't panic
2. **Dimension tests** - Output dimensions are correct
3. **Correctness tests** - Output is semantically correct
4. **Edge cases** - Small images, large images, different channel counts
5. **Parameter tests** - Different parameter combinations
6. **Backend tests** - CPU and GPU backends both work
7. **Parity tests** - Matches OpenCV.js behavior

### GPU Test Guidelines

GPU tests should:
- Check if GPU is available before asserting success
- Fail loudly if GPU is available but broken
- Skip gracefully if GPU is not supported
- Compare CPU and GPU results for consistency

Example:

```rust
#[wasm_bindgen_test]
async fn test_operation_gpu() {
    set_backend_wasm("webgpu");
    let result = operation(&input).await;

    match result {
        Ok(output) => {
            // GPU works - validate output
            assert_eq!(output.width(), expected_width);
        }
        Err(e) => {
            let msg = format!("{:?}", e);
            if msg.contains("GPU not available") {
                web_sys::console::log_1(&"GPU not available".into());
            } else {
                panic!("GPU failed: {:?}", e);
            }
        }
    }
}
```

## Debugging Failed Tests

### WASM Tests Failing Locally

1. **Run in interactive mode** to use browser DevTools:
   ```bash
   wasm-pack test --firefox --features wasm
   ```

2. **Check browser console** for error messages

3. **Add debug logging**:
   ```rust
   web_sys::console::log_1(&format!("Debug: {:?}", value).into());
   ```

### WASM Tests Passing Locally but Failing in CI

- **Browser differences**: CI uses headless browsers which may behave differently
- **GPU availability**: CI environments may not have WebGPU support
- **Timing issues**: Headless browsers may be slower

### Native Tests Failing

Standard Rust debugging applies:
```bash
# Run with output
cargo test -- --nocapture

# Run specific test with backtrace
RUST_BACKTRACE=1 cargo test test_name
```

## Performance Testing

Benchmarks are in `benches/`:

```bash
cargo bench
```

Results are saved to `target/criterion/`.

## Coverage

```bash
# Install tarpaulin (one-time)
cargo install cargo-tarpaulin

# Generate coverage report
cargo tarpaulin --out Html
```

## Troubleshooting

### "wasm-pack: command not found"

```bash
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
```

### "geckodriver not found"

```bash
# macOS
brew install geckodriver

# Ubuntu
wget https://github.com/mozilla/geckodriver/releases/download/v0.34.0/geckodriver-v0.34.0-linux64.tar.gz
tar -xzf geckodriver-v0.34.0-linux64.tar.gz
sudo mv geckodriver /usr/local/bin/
```

### "chromedriver not found"

```bash
# macOS
brew install chromedriver

# Ubuntu - see CI workflow for automated installation
```

### Tests timeout in browser

Increase timeout in test:
```rust
wasm_bindgen_test_configure!(run_in_browser);
// Timeout increased automatically for async tests
```

## Best Practices

1. **Run tests before pushing**: Install git hooks for automatic validation
2. **Test both backends**: Always test CPU and GPU paths
3. **Use descriptive test names**: Make failures self-documenting
4. **Keep tests fast**: Use small test images when possible
5. **Add regression tests**: When fixing bugs, add tests to prevent recurrence
6. **Document expected behavior**: Add comments explaining what tests verify

## Resources

- [wasm-bindgen-test Documentation](https://rustwasm.github.io/wasm-bindgen/wasm-bindgen-test/index.html)
- [wasm-pack Testing Guide](https://rustwasm.github.io/wasm-pack/book/commands/test.html)
- [WebGPU Specification](https://www.w3.org/TR/webgpu/)
- [OpenCV.js Documentation](https://docs.opencv.org/4.x/d5/d10/tutorial_js_root.html)
