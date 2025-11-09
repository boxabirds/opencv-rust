// Gaussian Blur Shader - Separable Implementation
// This shader implements horizontal and vertical passes for efficient Gaussian blurring

struct GaussianParams {
    width: u32,
    height: u32,
    channels: u32,
    kernel_size: u32,
    sigma: f32,
    _padding: array<u32, 3>, // Align to 16 bytes
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: GaussianParams;

// Precomputed Gaussian kernel weights (up to size 15)
@group(0) @binding(3) var<storage, read> kernel: array<f32>;

// Horizontal pass - blur along X axis
@compute @workgroup_size(16, 16)
fn gaussian_horizontal(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let half = params.kernel_size / 2u;
    let channels = params.channels;

    // Process each channel
    for (var ch = 0u; ch < channels; ch++) {
        var sum = 0.0;
        var weight_sum = 0.0;

        // Apply horizontal kernel
        for (var i = 0u; i < params.kernel_size; i++) {
            let offset = i32(i) - i32(half);
            let sample_x = clamp(i32(x) + offset, 0, i32(params.width) - 1);

            let idx = (u32(sample_x) + y * params.width) * channels + ch;
            let weight = kernel[i];

            sum += f32(input[idx]) * weight;
            weight_sum += weight;
        }

        let out_idx = (x + y * params.width) * channels + ch;
        output[out_idx] = u32(clamp(sum / weight_sum, 0.0, 255.0));
    }
}

// Vertical pass - blur along Y axis
@compute @workgroup_size(16, 16)
fn gaussian_vertical(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let half = params.kernel_size / 2u;
    let channels = params.channels;

    // Process each channel
    for (var ch = 0u; ch < channels; ch++) {
        var sum = 0.0;
        var weight_sum = 0.0;

        // Apply vertical kernel
        for (var i = 0u; i < params.kernel_size; i++) {
            let offset = i32(i) - i32(half);
            let sample_y = clamp(i32(y) + offset, 0, i32(params.height) - 1);

            let idx = (x + u32(sample_y) * params.width) * channels + ch;
            let weight = kernel[i];

            sum += f32(input[idx]) * weight;
            weight_sum += weight;
        }

        let out_idx = (x + y * params.width) * channels + ch;
        output[out_idx] = u32(clamp(sum / weight_sum, 0.0, 255.0));
    }
}
