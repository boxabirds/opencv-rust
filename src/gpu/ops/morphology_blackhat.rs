use crate::core::Mat;
use crate::error::Result;
use crate::gpu::ops::{morphology_closing_gpu_async, subtract_gpu_async};

/// Morphological black-hat operation (closing - src)
pub async fn morphology_blackhat_gpu_async(src: &Mat, dst: &mut Mat, ksize: i32) -> Result<()> {
    // BlackHat = Closing - Source
    let mut closed = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    // Perform closing
    morphology_closing_gpu_async(src, &mut closed, ksize).await?;

    // Subtract: closing - src
    subtract_gpu_async(&closed, src, dst).await?;

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn morphology_blackhat_gpu(src: &Mat, dst: &mut Mat, ksize: i32) -> Result<()> {
    pollster::block_on(morphology_blackhat_gpu_async(src, dst, ksize))
}
