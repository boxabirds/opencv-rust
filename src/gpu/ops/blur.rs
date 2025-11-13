#![allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss, clippy::cast_precision_loss)]
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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU-accelerated Gaussian blur using separable filter (async version)
pub async fn gaussian_blur_gpu_async(src: &Mat, dst: &mut Mat, ksize: Size, sigma: f64) -> Result<()> {
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

    let kernel_size = usize::try_from(ksize.width).unwrap_or(0);
    let kernel_weights = create_gaussian_kernel(kernel_size, sigma);

    // Execute both passes in a single GPU submission (intermediate buffer stays on GPU)
    execute_separable_blur(src, dst, &kernel_weights, sigma).await?;

    Ok(())
}

/// GPU-accelerated Gaussian blur using separable filter (sync wrapper for native)
#[cfg(not(target_arch = "wasm32"))]
pub fn gaussian_blur_gpu(src: &Mat, dst: &mut Mat, ksize: Size, sigma: f64) -> Result<()> {
    pollster::block_on(gaussian_blur_gpu_async(src, dst, ksize, sigma))
}

// Fixed-point scale (matches OpenCV's INTER_RESIZE_COEF_BITS)
const FIXED_POINT_BITS: u32 = 11;
const FIXED_POINT_SCALE: i32 = 1 << FIXED_POINT_BITS; // 2048

fn create_gaussian_kernel(size: usize, sigma: f64) -> Vec<i32> {
    #[allow(clippy::cast_precision_loss)]
    let sigma = if sigma <= 0.0 {
        0.3 * ((size as f64 - 1.0) * 0.5 - 1.0) + 0.8
    } else {
        sigma
    };

    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let half = (size / 2) as i32;
    let mut kernel_f64 = Vec::with_capacity(size);
    let mut sum = 0.0_f64;

    // Generate kernel in f64 for maximum precision
    for i in -half..=half {
        let x = f64::from(i);
        let exponent = -x * x / (2.0 * sigma * sigma);
        // Use libm::exp which works correctly in WASM
        let value = libm::exp(exponent);
        kernel_f64.push(value);
        sum += value;
    }

    // Normalize in f64
    for val in &mut kernel_f64 {
        *val /= sum;
    }

    // Convert to fixed-point i32 (scale by 2048 and round)
    let mut kernel_fixed = Vec::with_capacity(size);
    let mut sum_fixed = 0_i32;

    for &val in &kernel_f64 {
        #[allow(clippy::cast_possible_truncation)]
        let scaled = val * f64::from(FIXED_POINT_SCALE);
        let fixed = libm::round(scaled) as i32;
        kernel_fixed.push(fixed);
        sum_fixed += fixed;
    }

    // Adjust for rounding errors to ensure sum equals FIXED_POINT_SCALE exactly
    // This matches OpenCV's approach
    let diff = FIXED_POINT_SCALE - sum_fixed;
    if diff != 0 {
        // Add difference to center element (most significant weight)
        let center_idx = size / 2;
        kernel_fixed[center_idx] += diff;
    }

    kernel_fixed
}

async fn execute_separable_blur(
    src: &Mat,
    dst: &mut Mat,
    kernel_weights: &[i32],
    sigma: f64,
) -> Result<()> {
    // Get GPU context with platform-specific approach
    #[cfg(not(target_arch = "wasm32"))]
    let ctx = GpuContext::get()
        .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;

    let width = u32::try_from(src.cols()).unwrap_or(u32::MAX);
    let height = u32::try_from(src.rows()).unwrap_or(u32::MAX);
    let channels = u32::try_from(src.channels()).unwrap_or(u32::MAX);
    let kernel_size = u32::try_from(kernel_weights.len()).unwrap_or(u32::MAX);

    #[cfg(target_arch = "wasm32")]
    {
        // For WASM, clone device, queue, and adapter before async operations (they're Arc'd internally)
        let (device, queue, adapter) = GpuContext::with_gpu(|ctx| (ctx.device.clone(), ctx.queue.clone(), ctx.adapter.clone()))
            .ok_or_else(|| Error::GpuNotAvailable("GPU context not initialized".to_string()))?;

        let temp_ctx = GpuContext {
            device,
            queue,
            adapter,
        };

        return execute_separable_blur_impl(&temp_ctx, src, dst, kernel_weights, sigma, width, height, channels, kernel_size).await;
    }

    #[cfg(not(target_arch = "wasm32"))]
    return execute_separable_blur_impl(ctx, src, dst, kernel_weights, sigma, width, height, channels, kernel_size).await;
}

async fn execute_separable_blur_impl(
    ctx: &GpuContext,
    src: &Mat,
    dst: &mut Mat,
    kernel_weights: &[i32],
    sigma: f64,
    width: u32,
    height: u32,
    channels: u32,
    kernel_size: u32,
) -> Result<()> {
    // Create shader module (reused from cache on native, created fresh on WASM)
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

    // Create intermediate buffer for horizontal pass output (i32, 4 bytes per element - matches OpenCV C++)
    let intermediate_size = u64::from(width) * u64::from(height) * u64::from(channels) * 4;
    let intermediate_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Intermediate i32 Buffer"),
        size: intermediate_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // Create final output buffer (u8, 1 byte per element)
    let output_size = u64::from(width) * u64::from(height) * u64::from(channels);
    let output_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output u8 Buffer"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Create params buffer
    #[allow(clippy::cast_possible_truncation)]
    let params = GaussianParams {
        width,
        height,
        channels,
        kernel_size,
        sigma: sigma as f32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
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

    // Create bind group for horizontal pass (input → intermediate)
    let bind_group_h = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Horizontal Pass Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: intermediate_buffer.as_entire_binding(),
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

    // Create bind group for vertical pass (intermediate → output)
    let bind_group_v = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Vertical Pass Bind Group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: intermediate_buffer.as_entire_binding(),
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

    // Create pipeline layout
    let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Gaussian Blur Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Create compute pipelines for both passes
    let pipeline_h = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Horizontal Blur Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("gaussian_horizontal"),
        compilation_options: Default::default(),
        cache: None,
    });

    let pipeline_v = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Vertical Blur Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("gaussian_vertical"),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create command encoder and execute both compute passes
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Gaussian Blur Encoder"),
    });

    // Horizontal pass
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Horizontal Blur Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline_h);
        compute_pass.set_bind_group(0, &bind_group_h, &[]);
        compute_pass.dispatch_workgroups(
            width.div_ceil(16),
            height.div_ceil(16),
            1,
        );
    }

    // Vertical pass
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Vertical Blur Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(&pipeline_v);
        compute_pass.set_bind_group(0, &bind_group_v, &[]);
        compute_pass.dispatch_workgroups(
            width.div_ceil(16),
            height.div_ceil(16),
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
        pollster::block_on(receiver)
            .map_err(|_| Error::GpuError("Failed to receive buffer mapping result".to_string()))?
            .map_err(|e| Error::GpuError(format!("Buffer mapping failed: {e:?}")))?;
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
