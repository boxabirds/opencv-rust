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

    #[test]
    fn test_default_backend_is_auto() {
        assert_eq!(get_backend_name(), "auto");
    }

    #[test]
    fn test_set_backend() {
        set_backend("cpu");
        assert_eq!(get_backend_name(), "cpu");

        set_backend("webgpu");
        assert_eq!(get_backend_name(), "webgpu");

        set_backend("auto");
        assert_eq!(get_backend_name(), "auto");
    }

    #[test]
    fn test_invalid_backend_defaults_to_auto() {
        set_backend("invalid");
        assert_eq!(get_backend_name(), "auto");
    }

    #[test]
    fn test_get_backend_returns_number() {
        set_backend("cpu");
        assert_eq!(get_backend(), 2);

        set_backend("webgpu");
        assert_eq!(get_backend(), 1);

        set_backend("auto");
        let result = get_backend();
        // Should resolve to either GPU (1) or CPU (2)
        assert!(result == 1 || result == 2);
    }
}
