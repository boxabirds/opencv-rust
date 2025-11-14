// GPU gabor_filter - Auto-generated
// Template: filter_parallel

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
}

// Byte access helpers
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

// Constants for Gabor filter
const PI: f32 = 3.14159265359;
const SIGMA: f32 = 5.0;        // Gaussian envelope std dev
const THETA: f32 = 0.0;        // Orientation (radians)
const LAMBDA: f32 = 10.0;      // Wavelength
const GAMMA: f32 = 0.5;        // Spatial aspect ratio
const PSI: f32 = 0.0;          // Phase offset
const KERNEL_SIZE: i32 = 15;   // Must be odd

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.width + x;
    let pixel_base = idx * params.channels;

    let half_kernel = KERNEL_SIZE / 2;

    // Process each channel
    for (var c: u32 = 0u; c < params.channels; c++) {
        var sum: f32 = 0.0;
        var weight_sum: f32 = 0.0;

        // Apply Gabor kernel
        for (var ky: i32 = -half_kernel; ky <= half_kernel; ky++) {
            for (var kx: i32 = -half_kernel; kx <= half_kernel; kx++) {
                let px = i32(x) + kx;
                let py = i32(y) + ky;

                if (px >= 0 && px < i32(params.width) && py >= 0 && py < i32(params.height)) {
                    // Compute Gabor kernel value
                    let fx = f32(kx);
                    let fy = f32(ky);

                    // Rotate coordinates
                    let x_theta = fx * cos(THETA) + fy * sin(THETA);
                    let y_theta = -fx * sin(THETA) + fy * cos(THETA);

                    // Gaussian envelope
                    let gaussian = exp(-((x_theta * x_theta + GAMMA * GAMMA * y_theta * y_theta) / (2.0 * SIGMA * SIGMA)));

                    // Sinusoidal carrier
                    let sinusoid = cos(2.0 * PI * x_theta / LAMBDA + PSI);

                    // Gabor kernel value
                    let kernel_val = gaussian * sinusoid;

                    // Get pixel value
                    let pidx = u32(py) * params.width + u32(px);
                    let pb = pidx * params.channels + c;
                    let pixel_val = f32(read_byte(&input, pb));

                    sum += kernel_val * pixel_val;
                    weight_sum += abs(kernel_val);
                }
            }
        }

        // Normalize and clamp
        let result = select(128.0, sum / weight_sum + 128.0, weight_sum > 0.0001);

        write_byte(&output, pixel_base + c, u32(clamp(result, 0.0, 255.0)));
    }
}
