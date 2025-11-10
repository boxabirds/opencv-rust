// RGB to YCrCb color space conversion
// YCrCb is used in JPEG compression

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

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

    let r = f32(input[idx]);
    let g = f32(input[idx + 1u]);
    let b = f32(input[idx + 2u]);

    // ITU-R BT.601 conversion
    let yy = 0.299 * r + 0.587 * g + 0.114 * b;
    let cr = 0.713 * (r - yy) + 128.0;
    let cb = 0.564 * (b - yy) + 128.0;

    output[idx] = u32(clamp(yy, 0.0, 255.0));
    output[idx + 1u] = u32(clamp(cr, 0.0, 255.0));
    output[idx + 2u] = u32(clamp(cb, 0.0, 255.0));
}
