// Gaussian Blur Shader - Separable Implementation
// This shader implements horizontal and vertical passes for efficient Gaussian blurring

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

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: GaussianParams;

// Precomputed Gaussian kernel weights (up to size 15)
@group(0) @binding(3) var<storage, read> kernel: array<f32>;

// === Byte Access Helpers ===
// WebGPU buffers are u32-aligned, so RGBA bytes are packed into u32 words.
// These helpers extract individual bytes correctly.

fn read_byte(buffer: ptr<storage, array<u32>, read>, byte_index: u32) -> u32 {
    let u32_index = byte_index / 4u;
    let byte_offset = byte_index % 4u;
    let word = buffer[u32_index];
    return (word >> (byte_offset * 8u)) & 0xFFu;
}

fn write_byte(buffer: ptr<storage, array<u32>, read_write>, byte_index: u32, value: u32) {
    let u32_index = byte_index / 4u;
    let byte_offset = byte_index % 4u;
    let old_word = buffer[u32_index];
    let mask = ~(0xFFu << (byte_offset * 8u));
    let new_word = (old_word & mask) | ((value & 0xFFu) << (byte_offset * 8u));
    buffer[u32_index] = new_word;
}

// === End Byte Access Helpers ===

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

            let byte_idx = (u32(sample_x) + y * params.width) * channels + ch;
            let weight = kernel[i];

            sum += f32(read_byte(&input, byte_idx)) * weight;
            weight_sum += weight;
        }

        let out_byte_idx = (x + y * params.width) * channels + ch;
        write_byte(&output, out_byte_idx, u32(clamp(sum / weight_sum, 0.0, 255.0)));
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

            let byte_idx = (x + u32(sample_y) * params.width) * channels + ch;
            let weight = kernel[i];

            sum += f32(read_byte(&input, byte_idx)) * weight;
            weight_sum += weight;
        }

        let out_byte_idx = (x + y * params.width) * channels + ch;
        write_byte(&output, out_byte_idx, u32(clamp(sum / weight_sum, 0.0, 255.0)));
    }
}
