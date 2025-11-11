use crate::core::Mat;
use crate::error::Result;
use crate::gpu::ops::{morphology_opening_gpu_async, subtract_gpu_async};

/// Morphological top-hat operation (src - opening)
pub async fn morphology_tophat_gpu_async(src: &Mat, dst: &mut Mat, ksize: i32) -> Result<()> {
    // TopHat = Source - Opening
    let mut opened = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    // Perform opening
    morphology_opening_gpu_async(src, &mut opened, ksize).await?;

    // Subtract: src - opening
    subtract_gpu_async(src, &opened, dst).await?;

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn morphology_tophat_gpu(src: &Mat, dst: &mut Mat, ksize: i32) -> Result<()> {
    pollster::block_on(morphology_tophat_gpu_async(src, dst, ksize))
}
