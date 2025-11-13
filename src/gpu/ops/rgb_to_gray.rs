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
struct RgbToGrayParams {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
}

pub async fn rgb_to_gray_gpu_async(src: &Mat, dst: &mut Mat) -> Result<()> {
    if src.channels() != 3 && src.channels() != 4 {
        return Err(Error::InvalidParameter("RGB to Gray requires 3 or 4-channel input (RGB or RGBA)".to_string()));
    }
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation("GPU rgb_to_gray only supports U8 depth".to_string()));
    }

    // Output as 4-channel RGBA to avoid race conditions (gray replicated to RGB)
    *dst = Mat::new(src.rows(), src.cols(), 4, MatDepth::U8)?;

    #[cfg(target_arch = "wasm32")]
    {
        let (device, queue, adapter) = GpuContext::with_gpu(|ctx| {
            (ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone())
        })
        .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;
        let temp_ctx = GpuContext { device, queue, adapter };
        return execute_rgb_to_gray_impl(&temp_ctx, src, dst).await;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let ctx = GpuContext::get()
            .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;
        return execute_rgb_to_gray_impl(ctx, src, dst).await;
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn rgb_to_gray_gpu(src: &Mat, dst: &mut Mat) -> Result<()> {
    pollster::block_on(rgb_to_gray_gpu_async(src, dst))
}

async fn execute_rgb_to_gray_impl(ctx: &GpuContext, src: &Mat, dst: &mut Mat) -> Result<()> {
    let width = u32::try_from(src.cols()).unwrap_or(u32::MAX);
    let height = u32::try_from(src.rows()).unwrap_or(u32::MAX);
    let channels = u32::try_from(src.channels()).unwrap_or(u32::MAX);

    let input_data = src.data();
    let input_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: input_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Output as RGBA (4 channels) to avoid race conditions
    let output_buffer_size = u64::from(width) * u64::from(height) * 4;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = RgbToGrayParams { width, height, channels, _pad: 0 };
    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Use cached pipeline if available, otherwise error
    #[cfg(not(target_arch = "wasm32"))]
    let (bind_group_layout, compute_pipeline) = {
        let cached = PipelineCache::get_rgb_to_gray_pipeline()
            .ok_or_else(|| Error::GpuNotAvailable("Pipeline cache not initialized".to_string()))?;
        (&cached.bind_group_layout, &cached.compute_pipeline)
    };

    // Native: Execute with direct pipeline references
    #[cfg(not(target_arch = "wasm32"))]
    {
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RGB to Gray Bind Group"),
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RGB to Gray Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RGB to Gray Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_size = 16;
            let workgroup_count_x = width.div_ceil(workgroup_size);
            let workgroup_count_y = height.div_ceil(workgroup_size);
            compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
        }

        let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
        ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| { let _ = sender.send(result); });

        receiver.await.map_err(|_| Error::GpuError("Failed to receive map result".to_string()))?.map_err(|e| Error::GpuError(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        // Copy RGBA output
        dst.data_mut().copy_from_slice(&data[..]);
        drop(data);
        staging_buffer.unmap();
    }

    // WASM: Execute inside pipeline closure to avoid lifetime issues
    #[cfg(target_arch = "wasm32")]
    {
        PipelineCache::with_rgb_to_gray_pipeline(|cached| {
            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RGB to Gray Bind Group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: input_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: output_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
                ],
            });

            let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RGB to Gray Encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("RGB to Gray Compute Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&cached.compute_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let workgroup_size = 16;
                let workgroup_count_x = width.div_ceil(workgroup_size);
                let workgroup_count_y = height.div_ceil(workgroup_size);
                compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
            }

            ctx.queue.submit(Some(encoder.finish()));
            Ok::<(), Error>(())
        }).ok_or_else(|| Error::GpuNotAvailable("Pipeline cache not initialized".to_string()))??;

        let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Copy Encoder") });
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer_size);
        ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| { let _ = sender.send(result); });

        receiver.await.map_err(|_| Error::GpuError("Failed to receive map result".to_string()))?.map_err(|e| Error::GpuError(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        // Copy RGBA output
        dst.data_mut().copy_from_slice(&data[..]);
        drop(data);
        staging_buffer.unmap();
    }
    Ok(())
}
