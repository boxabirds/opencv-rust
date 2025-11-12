#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss, clippy::cast_precision_loss)]
use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};
use crate::gpu::device::GpuContext;
use wgpu;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct IntegralImageParams {
    width: u32,
    height: u32,
    pass: u32,
    _pad: u32,
}

pub async fn integral_image_gpu_async(src: &Mat, dst: &mut Mat) -> Result<()> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Integral image requires single-channel input".to_string(),
        ));
    }
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "GPU integral_image only supports U8 depth".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::F32)?;

    #[cfg(target_arch = "wasm32")]
    {
        let (device, queue, adapter) = GpuContext::with_gpu(|ctx| {
            (ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone())
        })
        .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;
        let temp_ctx = GpuContext { device, queue, adapter };
        return execute_integral_image_impl(&temp_ctx, src, dst).await;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let ctx = GpuContext::get()
            .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;
        return execute_integral_image_impl(ctx, src, dst).await;
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn integral_image_gpu(src: &Mat, dst: &mut Mat) -> Result<()> {
    pollster::block_on(integral_image_gpu_async(src, dst))
}

async fn execute_integral_image_impl(ctx: &GpuContext, src: &Mat, dst: &mut Mat) -> Result<()> {
    let width = u32::try_from(src.cols()).unwrap_or(u32::MAX);
    let height = u32::try_from(src.rows()).unwrap_or(u32::MAX);

    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("IntegralImage Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/integral_image.wgsl").into()),
    });

    let input_data = src.data();
    let input_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: input_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let output_buffer_size = (width * height * 4) as u64; // u32 = 4 bytes
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("IntegralImage Bind Group Layout"),
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

    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("IntegralImage Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Horizontal pass
    let params_h = IntegralImageParams {
        width,
        height,
        pass: 0,
        _pad: 0,
    };
    let params_h_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params H Buffer"),
        contents: bytemuck::bytes_of(&params_h),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_h = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("IntegralImage Bind Group H"),
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
                resource: params_h_buffer.as_entire_binding(),
            },
        ],
    });

    let horizontal_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("IntegralImage Horizontal Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("horizontal_scan"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("IntegralImage Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("IntegralImage Horizontal Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&horizontal_pipeline);
        compute_pass.set_bind_group(0, &bind_group_h, &[]);
        compute_pass.dispatch_workgroups(1, height, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
    // ctx.device.poll(wgpu::Maintain::Wait); // No longer needed in wgpu 27

    // Vertical pass
    let params_v = IntegralImageParams {
        width,
        height,
        pass: 1,
        _pad: 0,
    };
    let params_v_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params V Buffer"),
        contents: bytemuck::bytes_of(&params_v),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_v = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("IntegralImage Bind Group V"),
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
                resource: params_v_buffer.as_entire_binding(),
            },
        ],
    });

    let vertical_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("IntegralImage Vertical Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("vertical_scan"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("IntegralImage Encoder 2"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("IntegralImage Vertical Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&vertical_pipeline);
        compute_pass.set_bind_group(0, &bind_group_v, &[]);
        let workgroup_count_x = width.div_ceil(16);
        compute_pass.dispatch_workgroups(workgroup_count_x, 1, 1);
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
    // ctx.device.poll(wgpu::Maintain::Wait); // No longer needed in wgpu 27

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
