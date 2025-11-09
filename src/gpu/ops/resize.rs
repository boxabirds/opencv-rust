use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};
use crate::gpu::device::GpuContext;
use wgpu;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ResizeParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU-accelerated bilinear resize (async version)
pub async fn resize_gpu_async(src: &Mat, dst: &mut Mat, dst_width: usize, dst_height: usize) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "GPU resize only supports U8 depth".to_string(),
        ));
    }

    *dst = Mat::new(dst_height, dst_width, src.channels(), src.depth())?;

    execute_resize(src, dst).await
}

#[cfg(not(target_arch = "wasm32"))]
pub fn resize_gpu(src: &Mat, dst: &mut Mat, dst_width: usize, dst_height: usize) -> Result<()> {
    pollster::block_on(resize_gpu_async(src, dst, dst_width, dst_height))
}

async fn execute_resize(src: &Mat, dst: &mut Mat) -> Result<()> {
    // Get GPU context with platform-specific approach
    #[cfg(not(target_arch = "wasm32"))]
    let ctx = GpuContext::get()
        .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;

    let src_width = src.cols() as u32;
    let src_height = src.rows() as u32;
    let dst_width = dst.cols() as u32;
    let dst_height = dst.rows() as u32;
    let channels = src.channels() as u32;

    #[cfg(target_arch = "wasm32")]
    {
        // For WASM, clone device and queue before async operations (they're Arc'd internally)
        let (device, queue) = GpuContext::with_gpu(|ctx| (ctx.device.clone(), ctx.queue.clone()))
            .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;

        let temp_ctx = GpuContext {
            device,
            queue,
            adapter: unsafe { std::mem::zeroed() }, // Not needed for compute operations
        };

        return execute_resize_impl(&temp_ctx, src, dst, src_width, src_height, dst_width, dst_height, channels).await;
    }

    #[cfg(not(target_arch = "wasm32"))]
    return execute_resize_impl(ctx, src, dst, src_width, src_height, dst_width, dst_height, channels).await;
}

async fn execute_resize_impl(
    ctx: &GpuContext,
    src: &Mat,
    dst: &mut Mat,
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
) -> Result<()> {
    // Create shader module
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Resize Shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../shaders/resize.wgsl").into()
        ),
    });

    // Create input buffer
    let input_data = src.data();
    let input_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Resize Input Buffer"),
        contents: input_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create output buffer
    let output_size = (dst_width * dst_height * channels) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Resize Output Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create params buffer
    let params = ResizeParams {
        src_width,
        src_height,
        dst_width,
        dst_height,
        channels,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Resize Params Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group layout
    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Resize Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Create bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Resize Bind Group"),
        layout: &bind_group_layout,
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

    // Create compute pipeline
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Resize Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Resize Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("resize_bilinear"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create command encoder and execute
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Resize Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Resize Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            (dst_width + 15) / 16,
            (dst_height + 15) / 16,
            1,
        );
    }

    // Create staging buffer for readback
    let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Resize Staging Buffer"),
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
