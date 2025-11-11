//! Core image processing operations

pub mod threshold;
pub mod edge;
pub mod filtering;

// Re-export all public functions for WASM bindings
#[cfg(target_arch = "wasm32")]
pub use threshold::{threshold_wasm, adaptive_threshold_wasm};
#[cfg(target_arch = "wasm32")]
pub use edge::{canny_wasm, sobel_wasm, scharr_wasm, laplacian_wasm};
#[cfg(target_arch = "wasm32")]
pub use filtering::{
    gaussian_blur_wasm, blur_wasm, box_blur_wasm, median_blur_wasm,
    bilateral_filter_wasm, guided_filter_wasm, gabor_filter_wasm
};
