#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss, clippy::cast_precision_loss)]
use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};
use crate::gpu::device::GpuContext;
use wgpu;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GaborParams {
    width: u32,
    height: u32,
    channels: u32,
    kernel_size: u32,
    sigma: f32,
    theta: f32,
    lambda: f32,
    gamma: f32,
    psi: f32,
    _pad: [u32; 3],
}

pub async fn gabor_filter_gpu_async(
    src: &Mat,
    dst: &mut Mat,
    ksize: i32,
    sigma: f64,
    theta: f64,
    lambda: f64,
    gamma: f64,
    psi: f64,
) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "GPU gabor_filter only supports U8 depth".to_string(),
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
        return execute_gabor_filter_impl(&temp_ctx, src, dst, ksize, sigma, theta, lambda, gamma, psi).await;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let result = GpuContext::with_gpu(|ctx| {
            pollster::block_on(execute_gabor_filter_impl(ctx, src, dst, ksize, sigma, theta, lambda, gamma, psi))
        })
        .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))??;
        return Ok(result);
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn gabor_filter_gpu(
    src: &Mat,
    dst: &mut Mat,
    ksize: i32,
    sigma: f64,
    theta: f64,
    lambda: f64,
    gamma: f64,
    psi: f64,
) -> Result<()> {
    pollster::block_on(gabor_filter_gpu_async(src, dst, ksize, sigma, theta, lambda, gamma, psi))
}

async fn execute_gabor_filter_impl(
    ctx: &GpuContext,
    src: &Mat,
    dst: &mut Mat,
    ksize: i32,
    sigma: f64,
    theta: f64,
    lambda: f64,
    gamma: f64,
    psi: f64,
) -> Result<()> {
    let width = u32::try_from(src.cols()).unwrap_or(u32::MAX);
    let height = u32::try_from(src.rows()).unwrap_or(u32::MAX);
    let channels = u32::try_from(src.channels()).unwrap_or(u32::MAX);

    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Gabor Filter Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/gabor_filter.wgsl").into()),
    });

    let input_data = src.data();
    let input_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: input_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let output_buffer_size = u64::from(width) * u64::from(height) * u64::from(channels);
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = GaborParams {
        width,
        height,
        channels,
        kernel_size: ksize as u32,
        sigma: sigma as f32,
        theta: theta as f32,
        lambda: lambda as f32,
        gamma: gamma as f32,
        psi: psi as f32,
        _pad: [0; 3],
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Gabor Bind Group Layout"),
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
        label: Some("Gabor Bind Group"),
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
        label: Some("Gabor Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Gabor Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Gabor Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Gabor Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);
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
    let (tx, rx) = futures::channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });

    #[cfg(not(target_arch = "wasm32"))]
    ctx.device.poll(wgpu::MaintainBase::Wait);

    rx.await.unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();
    dst.data_mut().copy_from_slice(&data);
    drop(data);
    staging_buffer.unmap();

    Ok(())
}
