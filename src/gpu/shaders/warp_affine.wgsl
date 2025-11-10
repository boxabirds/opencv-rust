// Affine transformation shader
// Applies 2x3 affine matrix to transform image

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
    // Affine matrix [a b c; d e f]
    m00: f32,  // a
    m01: f32,  // b
    m02: f32,  // c
    m10: f32,  // d
    m11: f32,  // e
    m12: f32,  // f
    _pad0: f32,
    _pad1: f32,
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.dst_width || y >= params.dst_height) {
        return;
    }

    // Transform destination coordinates to source coordinates
    let fx = f32(x);
    let fy = f32(y);

    // Inverse transformation: dst -> src
    let src_x = params.m00 * fx + params.m01 * fy + params.m02;
    let src_y = params.m10 * fx + params.m11 * fy + params.m12;

    // Check bounds
    if (src_x < 0.0 || src_x >= f32(params.src_width) - 1.0 ||
        src_y < 0.0 || src_y >= f32(params.src_height) - 1.0) {
        // Out of bounds - write black
        let dst_idx = (y * params.dst_width + x) * params.channels;
        for (var c: u32 = 0u; c < params.channels; c++) {
            output[dst_idx + c] = 0u;
        }
        return;
    }

    // Bilinear interpolation
    let x0 = u32(floor(src_x));
    let y0 = u32(floor(src_y));
    let x1 = x0 + 1u;
    let y1 = y0 + 1u;

    let fx0 = src_x - floor(src_x);
    let fy0 = src_y - floor(src_y);
    let fx1 = 1.0 - fx0;
    let fy1 = 1.0 - fy0;

    let dst_idx = (y * params.dst_width + x) * params.channels;

    for (var c: u32 = 0u; c < params.channels; c++) {
        let v00 = f32(input[(y0 * params.src_width + x0) * params.channels + c]);
        let v10 = f32(input[(y0 * params.src_width + x1) * params.channels + c]);
        let v01 = f32(input[(y1 * params.src_width + x0) * params.channels + c]);
        let v11 = f32(input[(y1 * params.src_width + x1) * params.channels + c]);

        let value = v00 * fx1 * fy1 + v10 * fx0 * fy1 + v01 * fx1 * fy0 + v11 * fx0 * fy0;
        output[dst_idx + c] = u32(value);
    }
}
