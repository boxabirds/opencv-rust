use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};
use crate::gpu::device::GpuContext;
use wgpu;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct EqualizeHistParams {
    width: u32,
    height: u32,
    pass: u32,
    _pad: u32,
}

pub async fn equalize_hist_gpu_async(src: &Mat, dst: &mut Mat) -> Result<()> {
    if src.channels() != 1 {
        return Err(Error::InvalidParameter(
            "Histogram equalization requires single-channel input".to_string(),
        ));
    }
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "GPU equalize_hist only supports U8 depth".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), 1, MatDepth::U8)?;

    #[cfg(target_arch = "wasm32")]
    {
        let (device, queue, adapter) = GpuContext::with_gpu(|ctx| {
            (ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone())
        })
        .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;
        let temp_ctx = GpuContext { device, queue, adapter };
        return execute_equalize_hist_impl(&temp_ctx, src, dst).await;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let ctx = GpuContext::get()
            .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;
        return execute_equalize_hist_impl(ctx, src, dst).await;
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn equalize_hist_gpu(src: &Mat, dst: &mut Mat) -> Result<()> {
    pollster::block_on(equalize_hist_gpu_async(src, dst))
}

async fn execute_equalize_hist_impl(ctx: &GpuContext, src: &Mat, dst: &mut Mat) -> Result<()> {
    let width = src.cols() as u32;
    let height = src.rows() as u32;

    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("EqualizeHist Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/equalize_hist.wgsl").into()),
    });

    let input_data = src.data();
    let input_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: input_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let output_buffer_size = (width * height) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let histogram_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Histogram Buffer"),
        size: 256 * 4, // 256 u32 values
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let cdf_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("CDF Buffer"),
        size: 256 * 4, // 256 u32 values
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("EqualizeHist Bind Group Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
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
        label: Some("EqualizeHist Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Clear histogram buffer
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Clear Histogram Encoder"),
    });
    encoder.clear_buffer(&histogram_buffer, 0, None);
    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::MaintainBase::Wait);

    // Pass 0: Compute histogram
    let params_0 = EqualizeHistParams {
        width,
        height,
        pass: 0,
        _pad: 0,
    };
    let params_0_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params 0 Buffer"),
        contents: bytemuck::bytes_of(&params_0),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_0 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("EqualizeHist Bind Group 0"),
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
                resource: histogram_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cdf_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_0_buffer.as_entire_binding(),
            },
        ],
    });

    let histogram_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("EqualizeHist Histogram Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("compute_histogram"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("EqualizeHist Histogram Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("EqualizeHist Histogram Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&histogram_pipeline);
        compute_pass.set_bind_group(0, &bind_group_0, &[]);
        let workgroup_count_x = (width + 15) / 16;
        let workgroup_count_y = (height + 15) / 16;
        compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::MaintainBase::Wait);

    // Pass 1: Compute CDF
    let params_1 = EqualizeHistParams {
        width,
        height,
        pass: 1,
        _pad: 0,
    };
    let params_1_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params 1 Buffer"),
        contents: bytemuck::bytes_of(&params_1),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_1 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("EqualizeHist Bind Group 1"),
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
                resource: histogram_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cdf_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_1_buffer.as_entire_binding(),
            },
        ],
    });

    let cdf_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("EqualizeHist CDF Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("compute_cdf"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("EqualizeHist CDF Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("EqualizeHist CDF Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&cdf_pipeline);
        compute_pass.set_bind_group(0, &bind_group_1, &[]);
        compute_pass.dispatch_workgroups(1, 1, 1);
    }

    ctx.queue.submit(Some(encoder.finish()));
    ctx.device.poll(wgpu::MaintainBase::Wait);

    // Pass 2: Apply equalization
    let params_2 = EqualizeHistParams {
        width,
        height,
        pass: 2,
        _pad: 0,
    };
    let params_2_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params 2 Buffer"),
        contents: bytemuck::bytes_of(&params_2),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_2 = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("EqualizeHist Bind Group 2"),
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
                resource: histogram_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cdf_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: params_2_buffer.as_entire_binding(),
            },
        ],
    });

    let equalize_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("EqualizeHist Equalize Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("apply_equalization"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("EqualizeHist Equalize Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("EqualizeHist Equalize Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&equalize_pipeline);
        compute_pass.set_bind_group(0, &bind_group_2, &[]);
        let workgroup_count_x = (width + 15) / 16;
        let workgroup_count_y = (height + 15) / 16;
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
