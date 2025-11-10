// Add weighted: dst = src1 * alpha + src2 * beta + gamma

@group(0) @binding(0) var<storage, read> input1: array<u32>;
@group(0) @binding(1) var<storage, read> input2: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
    alpha: f32,
    beta: f32,
    gamma: f32,
    _pad2: f32,
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
        let value = f32(input1[idx + c]) * params.alpha +
                     f32(input2[idx + c]) * params.beta +
                     params.gamma;
        output[idx + c] = u32(clamp(value, 0.0, 255.0));
    }
}
