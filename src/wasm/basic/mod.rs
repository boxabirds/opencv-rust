//! Core image processing operations

pub mod threshold;
pub mod edge;
pub mod filtering;

// Re-export all public functions for WASM bindings
#[cfg(target_arch = "wasm32")]
pub use threshold::{threshold_wasm, adaptive_threshold_wasm};
#[cfg(target_arch = "wasm32")]
pub use edge::{canny_wasm, sobel_wasm, scharr_wasm, laplacian_wasm};
