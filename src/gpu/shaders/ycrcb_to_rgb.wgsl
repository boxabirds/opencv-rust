// YCrCb to RGB color space conversion

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

    let yy = f32(input[idx]);
    let cr = f32(input[idx + 1u]) - 128.0;
    let cb = f32(input[idx + 2u]) - 128.0;

    // ITU-R BT.601 inverse conversion
    let r = yy + 1.403 * cr;
    let g = yy - 0.714 * cr - 0.344 * cb;
    let b = yy + 1.773 * cb;

    output[idx] = u32(clamp(r, 0.0, 255.0));
    output[idx + 1u] = u32(clamp(g, 0.0, 255.0));
    output[idx + 2u] = u32(clamp(b, 0.0, 255.0));
}
