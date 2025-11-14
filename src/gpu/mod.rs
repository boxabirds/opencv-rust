//! GPU acceleration module using WebGPU
//!
//! This module provides GPU-accelerated implementations of image processing operations
//! using the wgpu library, which works on both native platforms (via Vulkan/Metal/DX12)
//! and in the browser (via WebGPU).
//!
//! # GPU Batch Processing
//!
//! For optimal performance, use `GpuBatch` to chain multiple operations without
//! intermediate CPU transfers:
//!
//! ```no_run
//! use opencv_rust::gpu::GpuBatch;
//! use opencv_rust::core::types::ColorConversionCode;
//!
//! let result = GpuBatch::new()
//!     .gaussian_blur(5, 1.5)
//!     .cvt_color(ColorConversionCode::RgbToGray)
//!     .canny(50.0, 150.0)
//!     .execute(&image)?;
//! # Ok::<(), opencv_rust::error::Error>(())
//! ```

pub mod device;
pub mod batch;
pub mod pipeline_cache;

#[cfg(feature = "gpu")]
pub mod ops;

#[cfg(feature = "gpu")]
pub mod optical_flow;

#[cfg(feature = "gpu")]
pub use device::GpuContext;

pub use batch::GpuBatch;
pub use pipeline_cache::PipelineCache;

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
