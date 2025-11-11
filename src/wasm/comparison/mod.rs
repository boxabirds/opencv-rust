//! Comparison and bitwise operations

pub mod bitwise;

#[cfg(target_arch = "wasm32")]
pub use bitwise::*;
