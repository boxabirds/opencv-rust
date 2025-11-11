//! Modular WASM bindings structure
//!
//! This module provides the same WASM API as the monolithic mod.rs,
//! but organized into logical submodules for better maintainability.
//!
//! Enable with: cargo build --target wasm32-unknown-unknown --features wasm,wasm_modular

pub mod filtering;
pub mod morphology;
pub mod color;

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

// Re-export all morphology functions
#[cfg(target_arch = "wasm32")]
pub use morphology::{
    erode_wasm,
    dilate_wasm,
    morphology_opening_wasm,
    morphology_closing_wasm,
    morphology_gradient_wasm,
    morphology_top_hat_wasm,
    morphology_black_hat_wasm,
    morphology_tophat_wasm,
    morphology_blackhat_wasm,
};

// Re-export all color conversion functions
#[cfg(target_arch = "wasm32")]
pub use color::{
    cvt_color_gray_wasm,
    cvt_color_hsv_wasm,
    cvt_color_lab_wasm,
    cvt_color_ycrcb_wasm,
    rgb_to_gray_wasm,
    rgb_to_hsv_wasm,
    rgb_to_lab_wasm,
    rgb_to_ycrcb_wasm,
    hsv_to_rgb_wasm,
    lab_to_rgb_wasm,
    ycrcb_to_rgb_wasm,
};
