use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};
use crate::gpu::device::GpuContext;
use crate::gpu::pipeline_cache::PipelineCache;
use wgpu;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ThresholdParams {
    width: u32,
    height: u32,
    channels: u32,
    threshold: u32,
    max_value: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU-accelerated binary threshold (async version)
pub async fn threshold_gpu_async(src: &Mat, dst: &mut Mat, thresh: u8, max_value: u8) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "GPU threshold only supports U8 depth".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    execute_threshold(src, dst, thresh, max_value).await
}

#[cfg(not(target_arch = "wasm32"))]
pub fn threshold_gpu(src: &Mat, dst: &mut Mat, thresh: u8, max_value: u8) -> Result<()> {
    pollster::block_on(threshold_gpu_async(src, dst, thresh, max_value))
}

async fn execute_threshold(
    src: &Mat,
    dst: &mut Mat,
    thresh: u8,
    max_value: u8,
) -> Result<()> {
    // Get GPU context with platform-specific approach
    #[cfg(not(target_arch = "wasm32"))]
    let ctx = GpuContext::get()
        .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;

    let width = u32::try_from(src.cols()).unwrap_or(u32::MAX);
    let height = u32::try_from(src.rows()).unwrap_or(u32::MAX);
    let channels = u32::try_from(src.channels()).unwrap_or(u32::MAX);

    #[cfg(target_arch = "wasm32")]
    {
        // For WASM, clone device, queue, and adapter before async operations (they're Arc'd internally)
        let (device, queue, adapter) = GpuContext::with_gpu(|ctx| (ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone()))
            .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;

        let temp_ctx = GpuContext {
            device,
            queue,
            adapter,
        };

        return execute_threshold_impl(&temp_ctx, src, dst, thresh, max_value, width, height, channels).await;
    }

    #[cfg(not(target_arch = "wasm32"))]
    return execute_threshold_impl(ctx, src, dst, thresh, max_value, width, height, channels).await;
}

async fn execute_threshold_impl(
    ctx: &GpuContext,
    src: &Mat,
    dst: &mut Mat,
    thresh: u8,
    max_value: u8,
    width: u32,
    height: u32,
    channels: u32,
) -> Result<()> {
    // Create input buffer
    let input_data = src.data();
    let input_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Threshold Input Buffer"),
        contents: input_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create output buffer
    let output_size = u64::from(width) * u64::from(height) * u64::from(channels);
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Threshold Output Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create params buffer
    let params = ThresholdParams {
        width,
        height,
        channels,
        threshold: thresh as u32,
        max_value: max_value as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Threshold Params Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Use cached pipeline if available, otherwise error
    #[cfg(not(target_arch = "wasm32"))]
    let (bind_group_layout, compute_pipeline) = {
        let cached = PipelineCache::get_threshold_pipeline()
            .ok_or_else(|| Error::GpuNotAvailable("Pipeline cache not initialized".to_string()))?;
        // Use pre-compiled cached pipeline (fast!)
        (&cached.bind_group_layout, &cached.compute_pipeline)
    };

    #[cfg(target_arch = "wasm32")]
    let (bind_group_layout, compute_pipeline) = {
        PipelineCache::with_threshold_pipeline(|cached| {
            // Use pre-compiled cached pipeline (fast!)
            (&cached.bind_group_layout, &cached.compute_pipeline)
        }).ok_or_else(|| Error::GpuNotAvailable("Pipeline cache not initialized".to_string()))?
    };

    // Create bind group with cached layout
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Threshold Bind Group"),
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

    // Create command encoder and execute
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Threshold Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Threshold Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            (width + 15) / 16,
            (height + 15) / 16,
            1,
        );
    }

    // Create staging buffer for readback
    let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Threshold Staging Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);

    // Submit commands
    ctx.queue.submit(Some(encoder.finish()));

    // Read back results
    let buffer_slice = staging_buffer.slice(..);

    #[cfg(not(target_arch = "wasm32"))]
    {
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        pollster::block_on(receiver)
            .map_err(|_| Error::GpuError("Failed to receive buffer mapping result".to_string()))?
            .map_err(|e| Error::GpuError(format!("Buffer mapping failed: {:?}", e)))?;
    }

    #[cfg(target_arch = "wasm32")]
    {
        // In WASM, properly await the buffer mapping
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });
        receiver.await
            .map_err(|_| Error::GpuError("Failed to receive buffer mapping result".to_string()))?
            .map_err(|e| Error::GpuError(format!("Buffer mapping failed: {:?}", e)))?;
    }

    // Copy data to output Mat
    let data = buffer_slice.get_mapped_range();
    dst.data_mut().copy_from_slice(&data);

    drop(data);
    staging_buffer.unmap();

    Ok(())
}
