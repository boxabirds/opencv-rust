//! Miscellaneous operations

pub mod various;

#[cfg(target_arch = "wasm32")]
pub use various::*;
