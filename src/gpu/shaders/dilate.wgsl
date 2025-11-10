// Dilate morphological operation shader

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
    var maxs = array<u32, 4>();  // Support up to 4 channels

    // Initialize with min value
    for (var c: u32 = 0u; c < params.channels; c++) {
        maxs[c] = 0u;
    }

    // Find maximum values in neighborhood (dilate)
    for (var ky: i32 = -half_k; ky <= half_k; ky++) {
        for (var kx: i32 = -half_k; kx <= half_k; kx++) {
            let py = clamp(i32(y) + ky, 0, i32(params.height) - 1);
            let px = clamp(i32(x) + kx, 0, i32(params.width) - 1);
            let idx = (u32(py) * params.width + u32(px)) * params.channels;

            for (var c: u32 = 0u; c < params.channels; c++) {
                maxs[c] = max(maxs[c], input[idx + c]);
            }
        }
    }

    // Write maximum values
    let out_idx = (y * params.width + x) * params.channels;
    for (var c: u32 = 0u; c < params.channels; c++) {
        output[out_idx + c] = maxs[c];
    }
}
