@group(0) @binding(0) var<storage, read> input_ch0: array<u32>;
@group(0) @binding(1) var<storage, read> input_ch1: array<u32>;
@group(0) @binding(2) var<storage, read> input_ch2: array<u32>;
@group(0) @binding(3) var<storage, read> input_ch3: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<u32>;
@group(0) @binding(5) var<uniform> params: Params;

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

    let src_idx = y * params.width + x;
    let dst_idx = (y * params.width + x) * params.channels;

    if (params.channels >= 1u) {
        output[dst_idx] = input_ch0[src_idx];
    }
    if (params.channels >= 2u) {
        output[dst_idx + 1u] = input_ch1[src_idx];
    }
    if (params.channels >= 3u) {
        output[dst_idx + 2u] = input_ch2[src_idx];
    }
    if (params.channels >= 4u) {
        output[dst_idx + 3u] = input_ch3[src_idx];
    }
}
