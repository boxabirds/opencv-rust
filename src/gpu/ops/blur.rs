use crate::core::{Mat, MatDepth};
use crate::core::types::Size;
use crate::error::{Error, Result};
use crate::gpu::device::GpuContext;
use wgpu;
use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GaussianParams {
    width: u32,
    height: u32,
    channels: u32,
    kernel_size: u32,
    sigma: f32,
    _padding: [u32; 3],
}

/// GPU-accelerated Gaussian blur using separable filter
pub fn gaussian_blur_gpu(src: &Mat, dst: &mut Mat, ksize: Size, sigma: f64) -> Result<()> {
    if src.depth() != MatDepth::U8 {
        return Err(Error::UnsupportedOperation(
            "GPU gaussian_blur only supports U8 depth".to_string(),
        ));
    }

    if ksize.width % 2 == 0 || ksize.height % 2 == 0 {
        return Err(Error::InvalidParameter(
            "Kernel size must be odd".to_string(),
        ));
    }

    if ksize.width != ksize.height {
        return Err(Error::InvalidParameter(
            "GPU gaussian_blur currently only supports square kernels".to_string(),
        ));
    }

    *dst = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    let kernel_size = ksize.width as usize;
    let kernel_weights = create_gaussian_kernel(kernel_size, sigma);

    // Execute horizontal and vertical passes using intermediate buffer
    // (avoids borrow checker issue with dst being both source and destination)
    let mut temp = Mat::new(src.rows(), src.cols(), src.channels(), src.depth())?;

    GpuContext::with_gpu(|ctx| {
        execute_blur_pass(ctx, src, &mut temp, &kernel_weights, sigma, true)?;
        execute_blur_pass(ctx, &temp, dst, &kernel_weights, sigma, false)?;
        Ok::<(), Error>(())
    })
    .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))??;

    Ok(())
}

fn create_gaussian_kernel(size: usize, sigma: f64) -> Vec<f32> {
    let sigma = if sigma <= 0.0 {
        0.3 * ((size as f64 - 1.0) * 0.5 - 1.0) + 0.8
    } else {
        sigma
    };

    let half = (size / 2) as i32;
    let mut kernel = Vec::with_capacity(size);
    let mut sum = 0.0;

    for i in -half..=half {
        let x = i as f64;
        let value = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(value as f32);
        sum += value;
    }

    // Normalize
    for val in &mut kernel {
        *val /= sum as f32;
    }

    kernel
}

fn execute_blur_pass(
    ctx: &GpuContext,
    src: &Mat,
    dst: &mut Mat,
    kernel_weights: &[f32],
    sigma: f64,
    is_horizontal: bool,
) -> Result<()> {
    let width = src.cols() as u32;
    let height = src.rows() as u32;
    let channels = src.channels() as u32;
    let kernel_size = kernel_weights.len() as u32;

    // Create shader module
    let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Gaussian Blur Shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../shaders/gaussian_blur.wgsl").into()
        ),
    });

    // Create input buffer from Mat
    let input_data = src.data();
    let input_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: input_data,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create output buffer
    let output_size = (width * height * channels) as u64;
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create params buffer
    let params = GaussianParams {
        width,
        height,
        channels,
        kernel_size,
        sigma: sigma as f32,
        _padding: [0; 3],
    };

    let params_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Params Buffer"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // Create kernel weights buffer
    let kernel_buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Kernel Buffer"),
        contents: bytemuck::cast_slice(kernel_weights),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });

    // Create bind group layout
    let bind_group_layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Gaussian Blur Bind Group Layout"),
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
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // Create bind group
    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Gaussian Blur Bind Group"),
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: kernel_buffer.as_entire_binding(),
            },
        ],
    });

    // Create compute pipeline
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Gaussian Blur Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let entry_point = if is_horizontal {
        "gaussian_horizontal"
    } else {
        "gaussian_vertical"
    };

    let compute_pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Gaussian Blur Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point,
    });

    // Create command encoder and execute compute pass
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Gaussian Blur Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Gaussian Blur Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(
            (width + 15) / 16,
            (height + 15) / 16,
            1,
        );
    }

    // Create staging buffer for readback
    let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy output to staging buffer
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
        ctx.device.poll(wgpu::Maintain::Poll);
        pollster::block_on(receiver)
            .map_err(|_| Error::GpuError("Failed to receive buffer mapping result".to_string()))?
            .map_err(|e| Error::GpuError(format!("Buffer mapping failed: {:?}", e)))?;
    }

    #[cfg(target_arch = "wasm32")]
    {
        // In WASM, we can use a simpler synchronous pattern
        buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
        ctx.device.poll(wgpu::Maintain::Poll);
    }

    // Copy data to output Mat
    let data = buffer_slice.get_mapped_range();
    dst.data_mut().copy_from_slice(&data);

    drop(data);
    staging_buffer.unmap();

    Ok(())
}
