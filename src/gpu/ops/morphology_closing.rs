#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss, clippy::cast_precision_loss)]
use crate::core::Mat;
use crate::error::Result;
use crate::gpu::ops::{erode_gpu_async, dilate_gpu_async};

/// Morphological closing operation (dilate then erode)
pub async fn morphology_closing_gpu_async(src: &Mat, dst: &mut Mat, ksize: i32) -> Result<()> {
    // Closing = Dilate then Erode
    let mut temp = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    // First dilate
    dilate_gpu_async(src, &mut temp, ksize).await?;

    // Then erode
    erode_gpu_async(&temp, dst, ksize).await?;

    Ok(())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn morphology_closing_gpu(src: &Mat, dst: &mut Mat, ksize: i32) -> Result<()> {
    pollster::block_on(morphology_closing_gpu_async(src, dst, ksize))
}
