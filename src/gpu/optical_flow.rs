use crate::core::Mat;
use crate::error::Result;
use crate::gpu::device::GpuContext;
use wgpu::util::DeviceExt;

const SHADER_SOURCE: &str = include_str!("shaders/optical_flow_farneback.wgsl");

pub async fn calc_optical_flow_farneback_gpu(
    ctx: &GpuContext,
    prev: &Mat,
    next: &Mat,
    block_size: u32,
    search_range: u32,
) -> Result<Mat> {
    // Both images must be grayscale
    assert_eq!(prev.channels(), 1);
    assert_eq!(next.channels(), 1);
    assert_eq!(prev.rows(), next.rows());
    assert_eq!(prev.cols(), next.cols());

    let width = prev.cols() as u32;
    let height = prev.rows() as u32;

    // Create shader module
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Farneback Optical Flow Shader"),
        source: wgpu::ShaderSource::Wgsl(SHADER_SOURCE.into()),
    });

    // Prepare input buffers
    let prev_size = (width * height * 4) as usize; // Align to u32
    let next_size = prev_size;
    let flow_size = (width * height * 2 * 4) as usize; // 2 f32 channels

    let prev_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Prev Frame Buffer"),
        contents: bytemuck::cast_slice(prev.data()),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let next_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Next Frame Buffer"),
        contents: bytemuck::cast_slice(next.data()),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    let flow_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Flow Buffer"),
        size: flow_size as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Params {
        width: u32,
        height: u32,
        block_size: u32,
        search_range: u32,
    }

    let params = Params {
        width,
        height,
        block_size,
        search_range,
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::cast_slice(&[params]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group layout
    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Farneback Bind Group Layout"),
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
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Farneback Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: prev_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: next_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: flow_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Farneback Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Farneback Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Execute compute pass
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Farneback Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Farneback Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_size_x = 16;
        let workgroup_size_y = 16;
        let num_workgroups_x = (width + workgroup_size_x - 1) / workgroup_size_x;
        let num_workgroups_y = (height + workgroup_size_y - 1) / workgroup_size_y;

        compute_pass.dispatch_workgroups(num_workgroups_x, num_workgroups_y, 1);
    }

    // Read back results
    let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: flow_size as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(&flow_buffer, 0, &staging_buffer, 0, flow_size as u64);
    ctx.queue.submit(Some(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = futures::channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });

    // Poll only on native - WASM handles this automatically
    #[cfg(not(target_arch = "wasm32"))]
    ctx.device.poll(wgpu::MaintainBase::Wait);

    rx.await.unwrap().unwrap();

    let data = buffer_slice.get_mapped_range();
    let flow_data: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging_buffer.unmap();

    // Convert f32 flow to Mat (2 channels)
    let mut result = Mat::new(height as usize, width as usize, 2, crate::core::MatDepth::F32)?;
    let result_data = result.data_mut();
    let result_f32 = unsafe {
        std::slice::from_raw_parts_mut(
            result_data.as_mut_ptr() as *mut f32,
            (width * height * 2) as usize,
        )
    };
    result_f32.copy_from_slice(&flow_data);

    Ok(result)
}
