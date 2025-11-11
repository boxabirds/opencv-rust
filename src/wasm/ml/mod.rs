//! Machine learning operations

pub mod classifiers;

#[cfg(target_arch = "wasm32")]
pub use classifiers::*;
