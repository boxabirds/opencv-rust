use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};
use crate::gpu::device::GpuContext;
use wgpu;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MaxParams {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
}

pub async fn max_gpu_async(src1: &Mat, src2: &Mat, dst: &mut Mat) -> Result<()> {
    if src1.depth() != MatDepth::U8 || src2.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation("GPU max only supports U8 depth".to_string()));
    }
    if src1.rows() != src2.rows() || src1.cols() != src2.cols() || src1.channels() != src2.channels() {
        return Err(Error::InvalidParameter("Input images must have same dimensions and channels".to_string()));
    }

    *dst = Mat::new(src1.rows(), src1.cols(), src1.channels(), src1.depth())?;

    #[cfg(target_arch = "wasm32")]
    {
        let (device, queue, adapter) = GpuContext::with_gpu(|ctx| {
            (ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone())
        })
        .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;
        let temp_ctx = GpuContext { device, queue, adapter };
        return execute_max_impl(&temp_ctx, src1, src2, dst).await;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let ctx = GpuContext::get()
            .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;
        return execute_max_impl(ctx, src1, src2, dst).await;
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn max_gpu(src1: &Mat, src2: &Mat, dst: &mut Mat) -> Result<()> {
    pollster::block_on(max_gpu_async(src1, src2, dst))
}

async fn execute_max_impl(ctx: &GpuContext, src1: &Mat, src2: &Mat, dst: &mut Mat) -> Result<()> {
    let width = src1.cols() as u32;
    let height = src1.rows() as u32;
    let channels = src1.channels() as u32;

    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Max Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/max.wgsl").into()),
    });

    let input1_data = src1.data();
    let input1_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input1 Buffer"),
        contents: input1_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let input2_data = src2.data();
    let input2_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input2 Buffer"),
        contents: input2_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let output_buffer_size = (width * height * channels) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let params = MaxParams { width, height, channels, _pad: 0 };
    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Max Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Max Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: input1_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: input2_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: output_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
        ],
    });

    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Max Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Max Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Max Encoder") });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Max Compute Pass"), timestamp_writes: None });
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
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| { let _ = sender.send(result); });
    ctx.device.poll(wgpu::MaintainBase::Wait);

    receiver.await.map_err(|_| Error::GpuError("Failed to receive map result".to_string()))?.map_err(|e| Error::GpuError(format!("Buffer mapping failed: {:?}", e)))?;

    { let data = buffer_slice.get_mapped_range(); dst.data_mut().copy_from_slice(&data[..]); }
    staging_buffer.unmap();
    Ok(())
}
