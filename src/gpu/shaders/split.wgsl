@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output_ch0: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_ch1: array<u32>;
@group(0) @binding(3) var<storage, read_write> output_ch2: array<u32>;
@group(0) @binding(4) var<storage, read_write> output_ch3: array<u32>;
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

    let src_idx = (y * params.width + x) * params.channels;
    let dst_idx = y * params.width + x;

    if (params.channels >= 1u) {
        output_ch0[dst_idx] = input[src_idx];
    }
    if (params.channels >= 2u) {
        output_ch1[dst_idx] = input[src_idx + 1u];
    }
    if (params.channels >= 3u) {
        output_ch2[dst_idx] = input[src_idx + 2u];
    }
    if (params.channels >= 4u) {
        output_ch3[dst_idx] = input[src_idx + 3u];
    }
}
