// Element-wise minimum: dst = min(src1, src2)

@group(0) @binding(0) var<storage, read> input1: array<u32>;
@group(0) @binding(1) var<storage, read> input2: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
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
        output[idx + c] = min(input1[idx + c], input2[idx + c]);
    }
}
