//! Arithmetic operations

pub mod ops;

#[cfg(target_arch = "wasm32")]
pub use ops::*;
