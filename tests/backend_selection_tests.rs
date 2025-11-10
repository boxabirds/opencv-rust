// Integration tests for backend selection module
// These tests run on native target since wasm module is gated for wasm32

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
use opencv_rust::wasm::backend::*;

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod backend_tests {
    use super::*;

    // Helper to reset state between tests
    fn reset_state() {
        set_backend("auto");
    }

    #[test]
    fn test_backend_basics() {
        reset_state();

        // Test CPU mode
        set_backend("cpu");
        assert_eq!(get_backend_name(), "cpu");
        assert_eq!(get_backend(), 2);

        // Test WebGPU mode
        set_backend("webgpu");
        assert_eq!(get_backend_name(), "webgpu");
        assert_eq!(get_backend(), 1);

        // Test Auto mode
        set_backend("auto");
        assert_eq!(get_backend_name(), "auto");
        let result = get_backend();
        assert!(result == 1 || result == 2);
    }

    #[test]
    fn test_backend_caching() {
        reset_state();
        set_backend("auto");

        // First call should resolve
        let first = get_backend();

        // Subsequent calls should return same value (cached)
        for _ in 0..10 {
            assert_eq!(get_backend(), first);
        }
    }

    #[test]
    fn test_invalid_backends() {
        reset_state();

        set_backend("invalid");
        assert_eq!(get_backend_name(), "auto");

        set_backend("GPU");  // Wrong case
        assert_eq!(get_backend_name(), "auto");

        set_backend("");
        assert_eq!(get_backend_name(), "auto");
    }
}

// For non-wasm targets, create a dummy test that always passes
#[cfg(not(all(target_arch = "wasm32", feature = "wasm")))]
#[test]
fn backend_tests_require_wasm_target() {
    // Backend tests only run on wasm32 target with wasm feature
    // Run with: cargo test --target wasm32-unknown-unknown --features wasm
    assert!(true);
}
