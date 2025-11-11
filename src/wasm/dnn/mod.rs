//! Deep neural network operations

pub mod network;

#[cfg(target_arch = "wasm32")]
pub use network::*;
