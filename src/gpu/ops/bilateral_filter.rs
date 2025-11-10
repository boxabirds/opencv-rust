use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};
use crate::gpu::device::GpuContext;
use wgpu;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BilateralFilterParams {
    width: u32,
    height: u32,
    channels: u32,
    kernel_size: u32,
    sigma_color: f32,
    sigma_space: f32,
    _pad0: f32,
    _pad1: f32,
}

pub async fn bilateral_filter_gpu_async(
    src: &Mat,
    dst: &mut Mat,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "GPU bilateral_filter only supports U8 depth".to_string(),
        ));
    }

    if d <= 0 || d % 2 == 0 {
        return Err(Error::InvalidParameter(
            "Kernel diameter must be positive and odd".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    #[cfg(target_arch = "wasm32")]
    {
        let (device, queue, adapter) = GpuContext::with_gpu(|ctx| {
            (ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone())
        })
        .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;
        let temp_ctx = GpuContext { device, queue, adapter };
        return execute_bilateral_filter_impl(&temp_ctx, src, dst, d, sigma_color, sigma_space).await;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let ctx = GpuContext::get()
            .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;
        return execute_bilateral_filter_impl(ctx, src, dst, d, sigma_color, sigma_space).await;
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn bilateral_filter_gpu(
    src: &Mat,
    dst: &mut Mat,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
) -> Result<()> {
    pollster::block_on(bilateral_filter_gpu_async(src, dst, d, sigma_color, sigma_space))
}

async fn execute_bilateral_filter_impl(
    ctx: &GpuContext,
    src: &Mat,
    dst: &mut Mat,
    d: i32,
    sigma_color: f64,
    sigma_space: f64,
) -> Result<()> {
    let width = src.cols() as u32;
    let height = src.rows() as u32;
    let channels = src.channels() as u32;

    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Bilateral Filter Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/bilateral_filter.wgsl").into()),
    });

    let input_data = src.data();
    let input_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: input_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let output_buffer_size = (width * height * channels) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = BilateralFilterParams {
        width,
        height,
        channels,
        kernel_size: d as u32,
        sigma_color: sigma_color as f32,
        sigma_space: sigma_space as f32,
        _pad0: 0.0,
        _pad1: 0.0,
    };
    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bilateral Filter Bind Group Layout"),
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

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bilateral Filter Bind Group"),
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

    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Bilateral Filter Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Bilateral Filter Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Bilateral Filter Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Bilateral Filter Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_size = 16;
        let workgroup_count_x = (width + workgroup_size - 1) / workgroup_size;
        let workgroup_count_y = (height + workgroup_size - 1) / workgroup_size;
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
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    ctx.device.poll(wgpu::MaintainBase::Wait);

    receiver
        .await
        .map_err(|_| Error::GpuError("Failed to receive map result".to_string()))?
        .map_err(|e| Error::GpuError(format!("Buffer mapping failed: {:?}", e)))?;

    {
        let data = buffer_slice.get_mapped_range();
        dst.data_mut().copy_from_slice(&data[..]);
    }
    staging_buffer.unmap();
    Ok(())
}
