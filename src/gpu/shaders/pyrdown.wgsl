// Gaussian Pyramid Down - downsample with Gaussian blur
// Output size is (width+1)/2 x (height+1)/2

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

// 5x5 Gaussian kernel (approximation)
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

    // Map output coordinates to input (sample at even positions)
    let src_x = x * 2u;
    let src_y = y * 2u;

    let dst_idx = (y * params.dst_width + x) * params.channels;

    for (var c: u32 = 0u; c < params.channels; c++) {
        var sum: f32 = 0.0;

        // Apply 5x5 Gaussian kernel
        for (var ky: i32 = -2; ky <= 2; ky++) {
            for (var kx: i32 = -2; kx <= 2; kx++) {
                let py = clamp(i32(src_y) + ky, 0, i32(params.src_height) - 1);
                let px = clamp(i32(src_x) + kx, 0, i32(params.src_width) - 1);
                let src_idx = (u32(py) * params.src_width + u32(px)) * params.channels + c;
                sum += f32(input[src_idx]) * KERNEL[ky + 2][kx + 2];
            }
        }

        output[dst_idx + c] = u32(sum / 256.0);  // Normalize by kernel sum
    }
}
