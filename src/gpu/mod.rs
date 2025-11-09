//! GPU acceleration module using WebGPU
//!
//! This module provides GPU-accelerated implementations of image processing operations
//! using the wgpu library, which works on both native platforms (via Vulkan/Metal/DX12)
//! and in the browser (via WebGPU).

pub mod device;

#[cfg(feature = "gpu")]
pub mod ops;

#[cfg(feature = "gpu")]
pub use device::GpuContext;

/// Initialize GPU context (native only - blocks)
#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub fn init_gpu() -> bool {
    GpuContext::init()
}

/// Check if GPU is available
pub fn gpu_available() -> bool {
    #[cfg(feature = "gpu")]
    {
        GpuContext::is_available()
    }
    #[cfg(not(feature = "gpu"))]
    {
        false
    }
}
