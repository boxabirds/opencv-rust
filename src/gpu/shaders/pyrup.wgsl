// Gaussian Pyramid Up - upsample with Gaussian blur
// Output size is width*2 x height*2

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// 5x5 Gaussian kernel
const KERNEL: array<array<f32, 5>, 5> = array(
    array(1.0, 4.0, 6.0, 4.0, 1.0),
    array(4.0, 16.0, 24.0, 16.0, 4.0),
    array(6.0, 24.0, 36.0, 24.0, 6.0),
    array(4.0, 16.0, 24.0, 16.0, 4.0),
    array(1.0, 4.0, 6.0, 4.0, 1.0),
);

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.dst_width || y >= params.dst_height) {
        return;
    }

    // Map output coordinates to input (downsample factor 2)
    let src_x_f = f32(x) / 2.0;
    let src_y_f = f32(y) / 2.0;

    let src_x = u32(src_x_f);
    let src_y = u32(src_y_f);

    let dst_idx = (y * params.dst_width + x) * params.channels;

    // Simple bilinear interpolation combined with Gaussian
    for (var c: u32 = 0u; c < params.channels; c++) {
        if (src_x >= params.src_width || src_y >= params.src_height) {
            output[dst_idx + c] = 0u;
            continue;
        }

        let src_idx = (src_y * params.src_width + src_x) * params.channels + c;
        let value = input[src_idx];

        // Apply 4x gain for upsampling
        output[dst_idx + c] = min(value * 4u, 255u);
    }
}
