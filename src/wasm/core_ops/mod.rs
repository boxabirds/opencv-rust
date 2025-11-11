//! Core image processing operations for WASM

pub mod threshold;

// Re-export functions for easy access
#[cfg(target_arch = "wasm32")]
pub use threshold::{threshold_wasm, adaptive_threshold_wasm};
