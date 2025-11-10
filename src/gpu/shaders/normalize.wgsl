// Normalize image to specified range
// dst = (src - min_val) * (max_out - min_out) / (max_val - min_val) + min_out

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
    alpha: f32,  // (max_out - min_out) / (max_val - min_val)
    beta: f32,   // min_out - min_val * alpha
    _pad2: f32,
    _pad3: f32,
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = (y * params.width + x) * params.channels;

    for (var c: u32 = 0u; c < params.channels; c++) {
        let value = f32(input[idx + c]) * params.alpha + params.beta;
        output[idx + c] = u32(clamp(value, 0.0, 255.0));
    }
}
