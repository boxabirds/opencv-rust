// GPU guided_filter - Auto-generated
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

// Guided Filter parameters
const RADIUS: i32 = 5;         // Window radius
const EPSILON: f32 = 0.01;     // Regularization parameter

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.width + x;
    let pixel_base = idx * params.channels;

    // Guided filter: assumes input is both guide and input (can be generalized)
    // For simplicity, we process each channel independently

    for (var c: u32 = 0u; c < params.channels; c++) {
        // Compute local statistics in window
        var mean_guide: f32 = 0.0;
        var mean_input: f32 = 0.0;
        var mean_guide_input: f32 = 0.0;
        var mean_guide_sq: f32 = 0.0;
        var count: f32 = 0.0;

        for (var dy: i32 = -RADIUS; dy <= RADIUS; dy++) {
            for (var dx: i32 = -RADIUS; dx <= RADIUS; dx++) {
                let px = i32(x) + dx;
                let py = i32(y) + dy;

                if (px >= 0 && px < i32(params.width) && py >= 0 && py < i32(params.height)) {
                    let pidx = u32(py) * params.width + u32(px);
                    let pb = pidx * params.channels + c;
                    let val = f32(read_byte(&input, pb));

                    mean_guide += val;
                    mean_input += val;
                    mean_guide_input += val * val;
                    mean_guide_sq += val * val;
                    count += 1.0;
                }
            }
        }

        if (count > 0.0) {
            mean_guide /= count;
            mean_input /= count;
            mean_guide_input /= count;
            mean_guide_sq /= count;
        }

        // Compute variance and covariance
        let var_guide = mean_guide_sq - mean_guide * mean_guide;
        let cov_guide_input = mean_guide_input - mean_guide * mean_input;

        // Compute linear coefficients a and b
        let a = cov_guide_input / (var_guide + EPSILON);
        let b = mean_input - a * mean_guide;

        // Compute output as weighted average of (a*guide + b) in window
        var output_val: f32 = 0.0;
        var output_count: f32 = 0.0;

        for (var dy: i32 = -RADIUS; dy <= RADIUS; dy++) {
            for (var dx: i32 = -RADIUS; dx <= RADIUS; dx++) {
                let px = i32(x) + dx;
                let py = i32(y) + dy;

                if (px >= 0 && px < i32(params.width) && py >= 0 && py < i32(params.height)) {
                    let pidx = u32(py) * params.width + u32(px);
                    let pb = pidx * params.channels + c;
                    let guide_val = f32(read_byte(&input, pb));

                    output_val += a * guide_val + b;
                    output_count += 1.0;
                }
            }
        }

        let result = if (output_count > 0.0) {
            output_val / output_count
        } else {
            f32(read_byte(&input, pixel_base + c))
        };

        write_byte(&output, pixel_base + c, u32(clamp(result, 0.0, 255.0)));
    }
}
