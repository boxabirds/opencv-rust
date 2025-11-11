use crate::core::Mat;
use crate::error::Result;
use crate::gpu::ops::{erode_gpu_async, dilate_gpu_async, subtract_gpu_async};

/// Morphological gradient operation (dilate - erode)
pub async fn morphology_gradient_gpu_async(src: &Mat, dst: &mut Mat, ksize: i32) -> Result<()> {
    // Gradient = Dilate - Erode
    let mut dilated = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;
    let mut eroded = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    // Dilate
    dilate_gpu_async(src, &mut dilated, ksize).await?;

    // Erode
    erode_gpu_async(src, &mut eroded, ksize).await?;

    // Subtract
    subtract_gpu_async(&dilated, &eroded, dst).await?;

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn morphology_gradient_gpu(src: &Mat, dst: &mut Mat, ksize: i32) -> Result<()> {
    pollster::block_on(morphology_gradient_gpu_async(src, dst, ksize))
}
