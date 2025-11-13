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

// Precomputed Gaussian kernel weights in fixed-point format (scaled by 2048)
@group(0) @binding(3) var<storage, read> kernel: array<i32>;

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

// BORDER_REFLECT: Reflect at borders (OpenCV.js actual behavior)
// Despite BORDER_DEFAULT=4 (REFLECT_101), OpenCV.js actually uses REFLECT
// Example: abcd|efgh -> dcba|abcdefgh|hgfe (edge duplicated)
fn reflect101_x(x: i32, width: i32) -> u32 {
    var sx = x;
    if (sx < 0) {
        sx = -sx - 1;
    } else if (sx >= width) {
        sx = 2 * width - sx - 1;
    }
    return u32(clamp(sx, 0, width - 1));
}

fn reflect101_y(y: i32, height: i32) -> u32 {
    var sy = y;
    if (sy < 0) {
        sy = -sy - 1;
    } else if (sy >= height) {
        sy = 2 * height - sy - 1;
    }
    return u32(clamp(sy, 0, height - 1));
}

// Fixed-point constants (must match Rust side)
const FIXED_POINT_BITS: u32 = 11u;
const FIXED_POINT_SCALE: i32 = 2048;  // 1 << 11
const FIXED_POINT_ROUND: i32 = 1024;  // 1 << (11-1)
const FIXED_POINT_ROUND_DOUBLE: i32 = 2097152;  // 1 << 21 (for double application)

// Write i32 to buffer (4 bytes)
fn write_i32(buffer: ptr<storage, array<u32>, read_write>, index: u32, value: i32) {
    buffer[index] = bitcast<u32>(value);
}

// Read i32 from buffer (4 bytes)
fn read_i32(buffer: ptr<storage, array<u32>, read>, index: u32) -> i32 {
    return bitcast<i32>(buffer[index]);
}

// Horizontal pass - blur along X axis
// Output: i32 values scaled by 2048 (matches OpenCV C++ implementation)
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
        var sum = 0i;

        // Apply horizontal kernel using fixed-point arithmetic
        for (var i = 0u; i < params.kernel_size; i++) {
            let offset = i32(i) - i32(half);
            let sample_x = reflect101_x(i32(x) + offset, i32(params.width));

            let byte_idx = (sample_x + y * params.width) * channels + ch;
            let pixel = i32(read_byte(&input, byte_idx));
            let weight = kernel[i];

            sum += pixel * weight;
        }

        // Store as i32 (scaled by 2048, NO rounding - matches OpenCV)
        let out_idx = (x + y * params.width) * channels + ch;
        write_i32(&output, out_idx, sum);
    }
}

// Vertical pass - blur along Y axis
// Input: i32 values scaled by 2048 (from horizontal pass)
// Output: u8 values (round once and clamp - matches OpenCV C++)
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
        var sum = 0i;

        // Apply vertical kernel to i32 intermediate values
        for (var i = 0u; i < params.kernel_size; i++) {
            let offset = i32(i) - i32(half);
            let sample_y = reflect101_y(i32(y) + offset, i32(params.height));

            // Read i32 value (scaled by 2048) from intermediate buffer
            let idx = (x + sample_y * params.width) * channels + ch;
            let pixel_scaled = read_i32(&input, idx);
            let weight = kernel[i];

            // pixel_scaled is scaled by 2048, weight is scaled by 2048
            // so sum is scaled by 2048^2 = 2^22
            sum += pixel_scaled * weight;
        }

        // Scale back by 2^22 with rounding (matches OpenCV)
        let result = (sum + FIXED_POINT_ROUND_DOUBLE) >> 22u;

        let out_byte_idx = (x + y * params.width) * channels + ch;
        write_byte(&output, out_byte_idx, u32(clamp(result, 0, 255)));
    }
}
