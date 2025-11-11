//! Feature detection and matching operations

pub mod detection;
pub mod object;

// Re-export all public functions for WASM bindings
#[cfg(target_arch = "wasm32")]
pub use detection::{
    harris_corners_wasm, good_features_to_track_wasm, fast_wasm,
    sift_wasm, orb_wasm, brisk_wasm, akaze_wasm, kaze_wasm
};
