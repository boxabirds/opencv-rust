use crate::core::Mat;
use crate::error::Result;
use crate::gpu::ops::{erode_gpu_async, dilate_gpu_async};

/// Morphological opening operation (erode then dilate)
pub async fn morphology_opening_gpu_async(src: &Mat, dst: &mut Mat, ksize: i32) -> Result<()> {
    // Opening = Erode then Dilate
    let mut temp = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    // First erode
    erode_gpu_async(src, &mut temp, ksize).await?;

    // Then dilate
    dilate_gpu_async(&temp, dst, ksize).await?;

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn morphology_opening_gpu(src: &Mat, dst: &mut Mat, ksize: i32) -> Result<()> {
    pollster::block_on(morphology_opening_gpu_async(src, dst, ksize))
}
