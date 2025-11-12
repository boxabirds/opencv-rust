#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss, clippy::cast_precision_loss)]
use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};
use crate::gpu::device::GpuContext;
use crate::gpu::pipeline_cache::PipelineCache;
use wgpu;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SobelParams {
    width: u32,
    height: u32,
    dx: u32,
    dy: u32,
}

/// GPU-accelerated Sobel edge detection (async version)
pub async fn sobel_gpu_async(
    src: &Mat,
    dst: &mut Mat,
    dx: i32,
    dy: i32,
) -> Result<()> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Sobel only works on single-channel images".to_string(),
        ));
    }

    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "GPU sobel only supports U8 depth".to_string(),
        ));
    }

    if dx == 0 && dy == 0 {
        return Err(Error::InvalidParameter(
            "At least one of dx or dy must be non-zero".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    // Get GPU context with platform-specific approach
    #[cfg(target_arch = "wasm32")]
    {
        let (device, queue, adapter) = GpuContext::with_gpu(|ctx| {
            (ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone())
        })
        .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;

        let temp_ctx = GpuContext {
            device,
            queue,
            adapter,
        };

        return execute_sobel_impl(&temp_ctx, src, dst, dx, dy).await;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let ctx = GpuContext::get()
            .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;
        return execute_sobel_impl(ctx, src, dst, dx, dy).await;
    }
}

/// GPU-accelerated Sobel (sync wrapper for native)
#[cfg(not(target_arch = "wasm32"))]
pub fn sobel_gpu(src: &Mat, dst: &mut Mat, dx: i32, dy: i32) -> Result<()> {
    pollster::block_on(sobel_gpu_async(src, dst, dx, dy))
}

async fn execute_sobel_impl(
    ctx: &GpuContext,
    src: &Mat,
    dst: &mut Mat,
    dx: i32,
    dy: i32,
) -> Result<()> {
    let width = u32::try_from(src.cols()).unwrap_or(u32::MAX);
    let height = u32::try_from(src.rows()).unwrap_or(u32::MAX);

    // Create input buffer from Mat
    let input_data = src.data();
    let input_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: input_data,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

    // Create output buffer
    let output_buffer_size = u64::from(width) * u64::from(height);
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create params buffer
    let params = SobelParams {
        width,
        height,
        dx: if dx > 0 { 1 } else { 0 },
        dy: if dy > 0 { 1 } else { 0 },
    };

    let params_buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

    // Use cached pipeline if available, otherwise error
    #[cfg(not(target_arch = "wasm32"))]
    let (bind_group_layout, compute_pipeline) = {
        let cached = PipelineCache::get_sobel_pipeline()
            .ok_or_else(|| Error::GpuNotAvailable("Pipeline cache not initialized".to_string()))?;
        (&cached.bind_group_layout, &cached.compute_pipeline)
    };

    #[cfg(target_arch = "wasm32")]
    let (bind_group_layout, compute_pipeline) = {
        PipelineCache::with_sobel_pipeline(|cached| {
            (&cached.bind_group_layout, &cached.compute_pipeline)
        }).ok_or_else(|| Error::GpuNotAvailable("Pipeline cache not initialized".to_string()))?
    };

    // Create bind group with cached layout
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Sobel Bind Group"),
        layout: bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Create command encoder and dispatch compute
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Sobel Encoder"),
        });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Sobel Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        // Dispatch with 16x16 workgroups
        let workgroup_size = 16;
        let workgroup_count_x = (width + workgroup_size - 1) / workgroup_size;
        let workgroup_count_y = (height + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
    }

    // Create staging buffer for readback
    let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);

    // Submit commands
    ctx.queue.submit(Some(encoder.finish()));

    // Read back results
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures::channel::oneshot::channel();

    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });

    // ctx.device.poll(wgpu::Maintain::Wait); // No longer needed in wgpu 27

    receiver
        .await
        .map_err(|_| Error::GpuError("Failed to receive map result".to_string()))?
        .map_err(|e| Error::GpuError(format!("Buffer mapping failed: {:?}", e)))?;

    // Copy data to output Mat
    {
        let data = buffer_slice.get_mapped_range();
        dst.data_mut().copy_from_slice(&data[..]);
    }

    staging_buffer.unmap();

    Ok(())
}
