//! Video analysis operations

pub mod tracking;

#[cfg(target_arch = "wasm32")]
pub use tracking::*;
