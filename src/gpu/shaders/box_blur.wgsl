// Box blur (mean filter) shader

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    kernel_size: u32,
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let half_k = i32(params.kernel_size) / 2;
    let kernel_area = f32(params.kernel_size * params.kernel_size);

    var sums = array<f32, 4>();  // Support up to 4 channels

    // Apply box filter
    for (var ky: i32 = -half_k; ky <= half_k; ky++) {
        for (var kx: i32 = -half_k; kx <= half_k; kx++) {
            let py = clamp(i32(y) + ky, 0, i32(params.height) - 1);
            let px = clamp(i32(x) + kx, 0, i32(params.width) - 1);
            let idx = (u32(py) * params.width + u32(px)) * params.channels;

            for (var c: u32 = 0u; c < params.channels; c++) {
                sums[c] += f32(input[idx + c]);
            }
        }
    }

    // Write averaged values
    let out_idx = (y * params.width + x) * params.channels;
    for (var c: u32 = 0u; c < params.channels; c++) {
        output[out_idx + c] = u32(sums[c] / kernel_area);
    }
}
