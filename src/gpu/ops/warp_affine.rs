use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};
use crate::gpu::device::GpuContext;
use wgpu;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WarpAffineParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
    m00: f32,
    m01: f32,
    m02: f32,
    m10: f32,
    m11: f32,
    m12: f32,
    _pad0: f32,
    _pad1: f32,
}

pub async fn warp_affine_gpu_async(
    src: &Mat,
    dst: &mut Mat,
    m: &[f32; 6],
    dst_size: (usize, usize),
) -> Result<()> {
    if src.depth() != MatDepth::U8 && src.depth() != MatDepth::F32 {
        return Err(Error::UnsupportedOperation(
            "GPU warp_affine only supports U8 and F32 depth".to_string(),
        ));
    }

    let (dst_height, dst_width) = dst_size;
    *dst = Mat::new(dst_height, dst_width, src.channels(), src.depth())?;

    #[cfg(target_arch = "wasm32")]
    {
        let (device, queue, adapter) = GpuContext::with_gpu(|ctx| {
            (ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone())
        })
        .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;
        let temp_ctx = GpuContext { device, queue, adapter };
        return execute_warp_affine_impl(&temp_ctx, src, dst, m).await;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let ctx = GpuContext::get()
            .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;
        return execute_warp_affine_impl(ctx, src, dst, m).await;
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn warp_affine_gpu(
    src: &Mat,
    dst: &mut Mat,
    m: &[f32; 6],
    dst_size: (usize, usize),
) -> Result<()> {
    pollster::block_on(warp_affine_gpu_async(src, dst, m, dst_size))
}

async fn execute_warp_affine_impl(
    ctx: &GpuContext,
    src: &Mat,
    dst: &mut Mat,
    m: &[f32; 6],
) -> Result<()> {
    let src_width = u32::try_from(src.cols()).unwrap_or(u32::MAX);
    let src_height = u32::try_from(src.rows()).unwrap_or(u32::MAX);
    let dst_width = u32::try_from(dst.cols()).unwrap_or(u32::MAX);
    let dst_height = u32::try_from(dst.rows()).unwrap_or(u32::MAX);
    let channels = u32::try_from(src.channels()).unwrap_or(u32::MAX);

    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("WarpAffine Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/warp_affine.wgsl").into()),
    });

    let input_data = src.data();
    let input_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: input_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let output_buffer_size = (dst_width * dst_height * channels) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = WarpAffineParams {
        src_width,
        src_height,
        dst_width,
        dst_height,
        channels,
        m00: m[0],
        m01: m[1],
        m02: m[2],
        m10: m[3],
        m11: m[4],
        m12: m[5],
        _pad0: 0.0,
        _pad1: 0.0,
    };
    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("WarpAffine Bind Group Layout"),
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
        label: Some("WarpAffine Bind Group"),
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
        label: Some("WarpAffine Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("WarpAffine Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("WarpAffine Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("WarpAffine Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_size = 16;
        let workgroup_count_x = (dst_width + workgroup_size - 1) / workgroup_size;
        let workgroup_count_y = (dst_height + workgroup_size - 1) / workgroup_size;
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
