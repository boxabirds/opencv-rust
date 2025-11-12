// Native tests for backend selection module
// This module makes backend selection testable on native targets

use std::sync::atomic::{AtomicU8, Ordering};

/// User-facing backend setting
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Auto = 0,
    WebGPU = 1,
    CPU = 2,
}

/// Cached backend resolution
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedBackend {
    Unresolved = 0,
    GPU = 1,
    CPU = 2,
}

static BACKEND: AtomicU8 = AtomicU8::new(Backend::Auto as u8);
static RESOLVED: AtomicU8 = AtomicU8::new(ResolvedBackend::Unresolved as u8);

pub fn get_backend() -> u8 {
    let backend = BACKEND.load(Ordering::Relaxed);
    match backend {
        1 => 1,
        2 => 2,
        0 => {
            let resolved = RESOLVED.load(Ordering::Relaxed);
            if resolved != 0 {
                return resolved;
            }
            #[cfg(feature = "gpu")]
            let result = if crate::gpu::device::GpuContext::is_available() { 1 } else { 2 };
            #[cfg(not(feature = "gpu"))]
            let result = 2;
            RESOLVED.store(result, Ordering::Relaxed);
            result
        }
        _ => 2,
    }
}

pub fn set_backend(backend_str: &str) {
    let value = match backend_str {
        "webgpu" | "gpu" => Backend::WebGPU as u8,
        "cpu" => Backend::CPU as u8,
        "auto" => {
            RESOLVED.store(ResolvedBackend::Unresolved as u8, Ordering::Relaxed);
            Backend::Auto as u8
        }
        _ => Backend::Auto as u8,
    };
    BACKEND.store(value, Ordering::Relaxed);
}

pub fn get_backend_name() -> &'static str {
    match BACKEND.load(Ordering::Relaxed) {
        0 => "auto",
        1 => "webgpu",
        2 => "cpu",
        _ => "auto",
    }
}

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
    use serial_test::serial;

    fn reset() {
        BACKEND.store(Backend::Auto as u8, Ordering::Relaxed);
        RESOLVED.store(ResolvedBackend::Unresolved as u8, Ordering::Relaxed);
    }

    #[test]
    #[serial]
    #[serial]
    fn test_default_is_auto() {
        reset();
        assert_eq!(get_backend_name(), "auto");
    }

    #[test]
    #[serial]
    #[serial]
    fn test_set_cpu() {
        reset();
        set_backend("cpu");
        assert_eq!(get_backend_name(), "cpu");
        assert_eq!(get_backend(), 2);
    }

    #[test]
    #[serial]
    #[serial]
    fn test_set_webgpu() {
        reset();
        set_backend("webgpu");
        assert_eq!(get_backend_name(), "webgpu");
        assert_eq!(get_backend(), 1);
    }

    #[test]
    #[serial]
    fn test_gpu_alias() {
        reset();
        set_backend("gpu");
        assert_eq!(get_backend_name(), "webgpu");
        assert_eq!(get_backend(), 1);
    }

    #[test]
    #[serial]
    fn test_auto_resolution() {
        reset();
        set_backend("auto");
        let result = get_backend();
        assert!(result == 1 || result == 2, "Auto should resolve to 1 or 2");
    }

    #[test]
    #[serial]
    fn test_invalid_defaults_to_auto() {
        reset();
        set_backend("invalid");
        assert_eq!(get_backend_name(), "auto");
    }

    #[test]
    #[serial]
    fn test_empty_string() {
        reset();
        set_backend("");
        assert_eq!(get_backend_name(), "auto");
    }

    #[test]
    #[serial]
    fn test_case_sensitive() {
        reset();
        set_backend("CPU");
        assert_eq!(get_backend_name(), "auto");
    }

    #[test]
    #[serial]
    fn test_switching() {
        reset();
        set_backend("cpu");
        assert_eq!(get_backend(), 2);
        set_backend("webgpu");
        assert_eq!(get_backend(), 1);
        set_backend("cpu");
        assert_eq!(get_backend(), 2);
    }

    #[test]
    #[serial]
    fn test_auto_caching() {
        reset();
        set_backend("auto");
        let first = get_backend();
        // Check resolution was cached
        let resolved = get_resolved_backend_name();
        if first == 1 {
            assert_eq!(resolved, "gpu");
        } else {
            assert_eq!(resolved, "cpu");
        }
        // Second call returns cached value
        assert_eq!(get_backend(), first);
        // Third call too
        assert_eq!(get_backend(), first);
    }

    #[test]
    #[serial]
    fn test_reset_cache_on_auto() {
        reset();
        set_backend("auto");
        let _ = get_backend();
        set_backend("cpu");
        set_backend("auto");
        assert_eq!(get_resolved_backend_name(), "unresolved");
        let _ = get_backend();
        let resolved = get_resolved_backend_name();
        assert!(resolved == "gpu" || resolved == "cpu");
    }

    #[test]
    #[serial]
    fn test_cpu_mode_no_resolution() {
        reset();
        set_backend("cpu");
        assert_eq!(get_backend(), 2);
        assert_eq!(get_backend(), 2);
        assert_eq!(get_resolved_backend_name(), "unresolved");
    }

    #[test]
    #[serial]
    fn test_webgpu_mode_no_resolution() {
        reset();
        set_backend("webgpu");
        assert_eq!(get_backend(), 1);
        assert_eq!(get_backend(), 1);
        assert_eq!(get_resolved_backend_name(), "unresolved");
    }

    #[test]
    #[serial]
    fn test_resolution_before_get() {
        reset();
        set_backend("auto");
        assert_eq!(get_resolved_backend_name(), "unresolved");
        let _ = get_backend();
        let resolved = get_resolved_backend_name();
        assert!(resolved == "gpu" || resolved == "cpu");
    }

    #[test]
    #[serial]
    fn test_rapid_switching() {
        reset();
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
    #[serial]
    fn test_concurrent_calls() {
        reset();
        set_backend("auto");
        let results: Vec<u8> = (0..100).map(|_| get_backend()).collect();
        let first = results[0];
        for result in results {
            assert_eq!(result, first, "All calls should return cached value");
        }
    }

    #[test]
    #[serial]
    fn test_enum_values() {
        assert_eq!(Backend::Auto as u8, 0);
        assert_eq!(Backend::WebGPU as u8, 1);
        assert_eq!(Backend::CPU as u8, 2);
        assert_eq!(ResolvedBackend::Unresolved as u8, 0);
        assert_eq!(ResolvedBackend::GPU as u8, 1);
        assert_eq!(ResolvedBackend::CPU as u8, 2);
    }

    #[test]
    #[serial]
    #[cfg(not(feature = "gpu"))]
    fn test_no_gpu_feature() {
        reset();
        set_backend("auto");
        assert_eq!(get_backend(), 2, "Without GPU feature, should resolve to CPU");
        assert_eq!(get_resolved_backend_name(), "cpu");
    }
}
