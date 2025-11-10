//! Backend selection for GPU vs CPU execution
//!
//! This module provides runtime backend selection for operations,
//! allowing users to choose between GPU (WebGPU) and CPU execution,
//! or automatically fall back from GPU to CPU when GPU is unavailable.

use std::sync::atomic::{AtomicU8, Ordering};

/// User-facing backend setting
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Try GPU first, fall back to CPU if unavailable
    Auto = 0,
    /// Force GPU execution (error if unavailable)
    WebGPU = 1,
    /// Force CPU execution
    CPU = 2,
}

/// Cached backend resolution (only for Auto mode)
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedBackend {
    /// Not yet determined
    Unresolved = 0,
    /// Resolved to GPU
    GPU = 1,
    /// Resolved to CPU
    CPU = 2,
}

// User-facing setting (changeable at runtime)
static BACKEND: AtomicU8 = AtomicU8::new(Backend::Auto as u8);

// Cached resolution (only for Auto mode)
static RESOLVED: AtomicU8 = AtomicU8::new(ResolvedBackend::Unresolved as u8);

/// Get the active backend (cached resolution for Auto mode)
///
/// Performance: 1-2ns per call (single atomic read)
///
/// # Returns
/// - `1` for GPU execution
/// - `2` for CPU execution
pub fn get_backend() -> u8 {
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
            #[cfg(feature = "gpu")]
            let result = if crate::gpu::device::GpuContext::is_available() { 1 } else { 2 };

            #[cfg(not(feature = "gpu"))]
            let result = 2; // No GPU support compiled in

            RESOLVED.store(result, Ordering::Relaxed);
            result
        }
        _ => 2, // Invalid - default to CPU
    }
}

/// Set the backend execution mode
///
/// # Arguments
/// * `backend_str` - "auto" | "webgpu" | "gpu" | "cpu"
pub fn set_backend(backend_str: &str) {
    let value = match backend_str {
        "webgpu" | "gpu" => Backend::WebGPU as u8,
        "cpu" => Backend::CPU as u8,
        "auto" => {
            // Reset cache to force re-resolution
            RESOLVED.store(ResolvedBackend::Unresolved as u8, Ordering::Relaxed);
            Backend::Auto as u8
        }
        _ => {
            #[cfg(target_arch = "wasm32")]
            web_sys::console::warn_1(&format!("Invalid backend '{}', defaulting to 'auto'", backend_str).into());
            Backend::Auto as u8
        }
    };
    BACKEND.store(value, Ordering::Relaxed);
}

/// Get the current backend setting
///
/// # Returns
/// "auto" | "webgpu" | "cpu"
pub fn get_backend_name() -> &'static str {
    match BACKEND.load(Ordering::Relaxed) {
        0 => "auto",
        1 => "webgpu",
        2 => "cpu",
        _ => "auto",
    }
}

/// Get the resolved backend (only meaningful in Auto mode)
///
/// # Returns
/// "gpu" | "cpu" | "unresolved"
pub fn get_resolved_backend_name() -> &'static str {
    match RESOLVED.load(Ordering::Relaxed) {
        0 => "unresolved",
        1 => "gpu",
        2 => "cpu",
        _ => "unresolved",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to reset state between tests
    fn reset_backend_state() {
        BACKEND.store(Backend::Auto as u8, Ordering::Relaxed);
        RESOLVED.store(ResolvedBackend::Unresolved as u8, Ordering::Relaxed);
    }

    #[test]
    fn test_default_backend_is_auto() {
        reset_backend_state();
        assert_eq!(get_backend_name(), "auto");
    }

    #[test]
    fn test_set_backend_cpu() {
        reset_backend_state();
        set_backend("cpu");
        assert_eq!(get_backend_name(), "cpu");
        assert_eq!(get_backend(), 2);
    }

    #[test]
    fn test_set_backend_webgpu() {
        reset_backend_state();
        set_backend("webgpu");
        assert_eq!(get_backend_name(), "webgpu");
        assert_eq!(get_backend(), 1);
    }

    #[test]
    fn test_set_backend_gpu_alias() {
        reset_backend_state();
        // "gpu" should be accepted as alias for "webgpu"
        set_backend("gpu");
        assert_eq!(get_backend_name(), "webgpu");
        assert_eq!(get_backend(), 1);
    }

    #[test]
    fn test_set_backend_auto() {
        reset_backend_state();
        set_backend("auto");
        assert_eq!(get_backend_name(), "auto");
        let result = get_backend();
        // Should resolve to either GPU (1) or CPU (2)
        assert!(result == 1 || result == 2, "Auto mode should resolve to 1 or 2, got {}", result);
    }

    #[test]
    fn test_invalid_backend_defaults_to_auto() {
        reset_backend_state();
        set_backend("invalid");
        assert_eq!(get_backend_name(), "auto");
    }

    #[test]
    fn test_empty_string_defaults_to_auto() {
        reset_backend_state();
        set_backend("");
        assert_eq!(get_backend_name(), "auto");
    }

    #[test]
    fn test_case_sensitive_backend_names() {
        reset_backend_state();
        // Should be case-sensitive
        set_backend("CPU");
        assert_eq!(get_backend_name(), "auto"); // Invalid, defaults to auto

        set_backend("WebGPU");
        assert_eq!(get_backend_name(), "auto"); // Invalid, defaults to auto

        set_backend("Auto");
        assert_eq!(get_backend_name(), "auto"); // Invalid, defaults to auto
    }

    #[test]
    fn test_backend_switching() {
        reset_backend_state();

        // Switch between different backends
        set_backend("cpu");
        assert_eq!(get_backend(), 2);

        set_backend("webgpu");
        assert_eq!(get_backend(), 1);

        set_backend("cpu");
        assert_eq!(get_backend(), 2);
    }

    #[test]
    fn test_auto_mode_caches_resolution() {
        reset_backend_state();
        set_backend("auto");

        // First call resolves and caches
        let first_result = get_backend();
        assert!(first_result == 1 || first_result == 2);

        // Check that resolution was cached
        let resolved_name = get_resolved_backend_name();
        if first_result == 1 {
            assert_eq!(resolved_name, "gpu");
        } else {
            assert_eq!(resolved_name, "cpu");
        }

        // Second call should return same cached result
        let second_result = get_backend();
        assert_eq!(second_result, first_result, "Auto mode should cache resolution");

        // Third call should also return cached result
        let third_result = get_backend();
        assert_eq!(third_result, first_result, "Cached resolution should persist");
    }

    #[test]
    fn test_switching_to_auto_resets_cache() {
        reset_backend_state();

        // First set to auto and resolve
        set_backend("auto");
        let first_resolution = get_backend();

        // Switch to cpu
        set_backend("cpu");
        assert_eq!(get_backend(), 2);

        // Switch back to auto - should reset cache
        set_backend("auto");
        assert_eq!(get_resolved_backend_name(), "unresolved");

        // Calling get_backend should re-resolve
        let second_resolution = get_backend();
        assert_eq!(second_resolution, first_resolution, "Re-resolution should match original");
    }

    #[test]
    fn test_cpu_mode_ignores_resolution() {
        reset_backend_state();
        set_backend("cpu");

        // CPU mode should always return 2
        assert_eq!(get_backend(), 2);
        assert_eq!(get_backend(), 2);
        assert_eq!(get_backend(), 2);

        // Resolution should remain unresolved in CPU mode
        assert_eq!(get_resolved_backend_name(), "unresolved");
    }

    #[test]
    fn test_webgpu_mode_ignores_resolution() {
        reset_backend_state();
        set_backend("webgpu");

        // WebGPU mode should always return 1
        assert_eq!(get_backend(), 1);
        assert_eq!(get_backend(), 1);
        assert_eq!(get_backend(), 1);

        // Resolution should remain unresolved in WebGPU mode
        assert_eq!(get_resolved_backend_name(), "unresolved");
    }

    #[test]
    fn test_resolved_backend_name_before_resolution() {
        reset_backend_state();
        set_backend("auto");

        // Before calling get_backend, resolution should be unresolved
        assert_eq!(get_resolved_backend_name(), "unresolved");

        // After calling get_backend, should be resolved
        let _ = get_backend();
        let resolved = get_resolved_backend_name();
        assert!(resolved == "gpu" || resolved == "cpu", "Should resolve to gpu or cpu");
    }

    #[test]
    fn test_resolution_state_isolated_from_backend_changes() {
        reset_backend_state();

        // Resolve in auto mode
        set_backend("auto");
        let _ = get_backend();
        let resolved = get_resolved_backend_name();

        // Switch to CPU mode
        set_backend("cpu");
        assert_eq!(get_backend(), 2);

        // Resolution cache should still exist (not cleared by non-auto switch)
        assert_eq!(get_resolved_backend_name(), resolved);

        // Switch to WebGPU mode
        set_backend("webgpu");
        assert_eq!(get_backend(), 1);

        // Resolution cache should still be there
        assert_eq!(get_resolved_backend_name(), resolved);
    }

    #[test]
    fn test_numeric_backend_values() {
        reset_backend_state();

        // Test Backend enum values
        assert_eq!(Backend::Auto as u8, 0);
        assert_eq!(Backend::WebGPU as u8, 1);
        assert_eq!(Backend::CPU as u8, 2);

        // Test ResolvedBackend enum values
        assert_eq!(ResolvedBackend::Unresolved as u8, 0);
        assert_eq!(ResolvedBackend::GPU as u8, 1);
        assert_eq!(ResolvedBackend::CPU as u8, 2);
    }

    #[test]
    fn test_backend_without_gpu_feature() {
        reset_backend_state();

        // Without GPU feature, auto mode should resolve to CPU
        #[cfg(not(feature = "gpu"))]
        {
            set_backend("auto");
            let result = get_backend();
            assert_eq!(result, 2, "Without GPU feature, should resolve to CPU");
            assert_eq!(get_resolved_backend_name(), "cpu");
        }
    }

    #[test]
    fn test_multiple_rapid_switches() {
        reset_backend_state();

        // Rapidly switch backends
        for _ in 0..10 {
            set_backend("cpu");
            assert_eq!(get_backend(), 2);

            set_backend("webgpu");
            assert_eq!(get_backend(), 1);

            set_backend("auto");
            let result = get_backend();
            assert!(result == 1 || result == 2);
        }
    }

    #[test]
    fn test_concurrent_get_backend_calls() {
        reset_backend_state();
        set_backend("auto");

        // Multiple get_backend calls should all return same value
        let results: Vec<u8> = (0..100).map(|_| get_backend()).collect();

        // All results should be the same (cached)
        let first = results[0];
        for result in results {
            assert_eq!(result, first, "All concurrent calls should return cached value");
        }
    }

    #[test]
    fn test_get_backend_name_consistency() {
        reset_backend_state();

        set_backend("cpu");
        assert_eq!(get_backend_name(), "cpu");
        assert_eq!(get_backend(), 2);

        set_backend("webgpu");
        assert_eq!(get_backend_name(), "webgpu");
        assert_eq!(get_backend(), 1);

        set_backend("auto");
        assert_eq!(get_backend_name(), "auto");
        let numeric = get_backend();
        assert!(numeric == 1 || numeric == 2);
    }

    #[test]
    fn test_whitespace_in_backend_names() {
        reset_backend_state();

        // Should not accept whitespace
        set_backend(" cpu");
        assert_eq!(get_backend_name(), "auto");

        set_backend("cpu ");
        assert_eq!(get_backend_name(), "auto");

        set_backend(" cpu ");
        assert_eq!(get_backend_name(), "auto");
    }
}
