@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> lut: array<u32>;
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

    for (var c = 0u; c < params.channels; c++) {
        let idx = (y * params.width + x) * params.channels + c;
        let value = input[idx];
        // LUT has 256 entries per channel
        let lut_idx = c * 256u + value;
        output[idx] = lut[lut_idx];
    }
}
