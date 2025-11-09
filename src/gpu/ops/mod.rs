pub mod blur;
pub mod resize;
pub mod threshold;
pub mod canny;

// Export sync versions for native
#[cfg(not(target_arch = "wasm32"))]
pub use blur::gaussian_blur_gpu;
#[cfg(not(target_arch = "wasm32"))]
pub use resize::resize_gpu;
#[cfg(not(target_arch = "wasm32"))]
pub use threshold::threshold_gpu;
#[cfg(not(target_arch = "wasm32"))]
pub use canny::canny_gpu;

// Export async versions for WASM
pub use blur::gaussian_blur_gpu_async;
pub use resize::resize_gpu_async;
pub use threshold::threshold_gpu_async;
pub use canny::canny_gpu_async;
