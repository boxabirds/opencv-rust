//! Image processing operations

pub mod morphology;
pub mod color;
pub mod drawing;
pub mod geometric;
pub mod histogram;
pub mod contour;

// Re-export all public functions for WASM bindings
#[cfg(target_arch = "wasm32")]
pub use morphology::{
    erode_wasm, dilate_wasm, morphology_opening_wasm, morphology_closing_wasm,
    morphology_gradient_wasm, morphology_top_hat_wasm, morphology_black_hat_wasm,
    morphology_tophat_wasm, morphology_blackhat_wasm
};
