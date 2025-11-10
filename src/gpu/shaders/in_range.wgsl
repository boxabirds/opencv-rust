@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
    lower_b: u32,
    lower_g: u32,
    lower_r: u32,
    lower_a: u32,
    upper_b: u32,
    upper_g: u32,
    upper_r: u32,
    upper_a: u32,
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx_base = (y * params.width + x) * params.channels;

    var in_range = true;

    if (params.channels == 1u) {
        let val = input[idx_base];
        in_range = val >= params.lower_b && val <= params.upper_b;
    } else if (params.channels == 3u) {
        let b = input[idx_base];
        let g = input[idx_base + 1u];
        let r = input[idx_base + 2u];
        in_range = b >= params.lower_b && b <= params.upper_b &&
                   g >= params.lower_g && g <= params.upper_g &&
                   r >= params.lower_r && r <= params.upper_r;
    } else if (params.channels == 4u) {
        let b = input[idx_base];
        let g = input[idx_base + 1u];
        let r = input[idx_base + 2u];
        let a = input[idx_base + 3u];
        in_range = b >= params.lower_b && b <= params.upper_b &&
                   g >= params.lower_g && g <= params.upper_g &&
                   r >= params.lower_r && r <= params.upper_r &&
                   a >= params.lower_a && a <= params.upper_a;
    }

    let out_idx = y * params.width + x;
    output[out_idx] = select(0u, 255u, in_range);
}
