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
#[cfg(target_arch = "wasm32")]
pub use drawing::{
    draw_line_wasm, draw_rectangle_wasm, draw_circle_wasm,
    draw_ellipse_wasm, draw_polylines_wasm, put_text_wasm
};
#[cfg(target_arch = "wasm32")]
pub use geometric::{
    resize_wasm, flip_wasm, rotate_wasm, warp_affine_wasm,
    warp_perspective_wasm, get_rotation_matrix_2d_wasm, remap_wasm
};
