use crate::core::{Mat, MatDepth};
use crate::error::{Error, Result};
use crate::gpu::device::GpuContext;
use wgpu;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SplitParams {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
}

pub async fn split_gpu_async(src: &Mat, dst: &mut Vec<Mat>) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "GPU split only supports U8 depth".to_string(),
        ));
    }

    let channels = src.channels();
    if channels < 1 || channels > 4 {
        return Err(Error::InvalidParameter(
            "Split supports 1-4 channels".to_string(),
        ));
    }

    // Create output matrices
    dst.clear();
    for _ in 0..channels {
        dst.push(Mat::new(src.rows(), src.cols(), 1, src.depth())?);
    }

    #[cfg(target_arch = "wasm32")]
    {
        let (device, queue, adapter) = GpuContext::with_gpu(|ctx| {
            (ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone())
        })
        .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;
        let temp_ctx = GpuContext { device, queue, adapter };
        return execute_split_impl(&temp_ctx, src, dst).await;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let ctx = GpuContext::get()
            .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;
        return execute_split_impl(ctx, src, dst).await;
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub fn split_gpu(src: &Mat, dst: &mut Vec<Mat>) -> Result<()> {
    pollster::block_on(split_gpu_async(src, dst))
}

async fn execute_split_impl(ctx: &GpuContext, src: &Mat, dst: &mut Vec<Mat>) -> Result<()> {
    let width = src.cols() as u32;
    let height = src.rows() as u32;
    let channels = src.channels() as u32;

    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Split Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/split.wgsl").into()),
    });

    let input_data = src.data();
    let input_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: input_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let single_channel_size = (width * height) as u64;

    // Create output buffers for each channel (max 4)
    let mut output_buffers = Vec::new();
    for i in 0..4 {
        output_buffers.push(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Output Buffer {}", i)),
            size: single_channel_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
    }

    let params = SplitParams {
        width,
        height,
        channels,
        _pad: 0,
    };
    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Split Bind Group Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
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
        label: Some("Split Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffers[0].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_buffers[1].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffers[2].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: output_buffers[3].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Split Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Split Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Split Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Split Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        let workgroup_size = 16;
        let workgroup_count_x = (width + workgroup_size - 1) / workgroup_size;
        let workgroup_count_y = (height + workgroup_size - 1) / workgroup_size;
        compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
    }

    // Copy each channel to staging buffer and read back
    for i in 0..channels as usize {
        let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Staging Buffer {}", i)),
            size: single_channel_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&output_buffers[i], 0, &staging_buffer, 0, single_channel_size);
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
            dst[i].data_mut().copy_from_slice(&data[..]);
        }
        staging_buffer.unmap();

        // Create new encoder for next channel
        if i < channels as usize - 1 {
            encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Split Encoder"),
            });
        }
    }

    Ok(())
}
