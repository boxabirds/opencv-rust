//! Modular WASM bindings structure
//!
//! This module provides the same WASM API as the monolithic mod.rs,
//! but organized into logical submodules for better maintainability.
//!
//! Enable with: cargo build --target wasm32-unknown-unknown --features wasm,wasm_modular

pub mod filtering;

// Re-export all filtering functions
#[cfg(target_arch = "wasm32")]
pub use filtering::{
    gaussian_blur_wasm,
    blur_wasm,
    box_blur_wasm,
    median_blur_wasm,
    bilateral_filter_wasm,
    laplacian_wasm,
    guided_filter_wasm,
    gabor_filter_wasm,
    filter2d_wasm,
};
