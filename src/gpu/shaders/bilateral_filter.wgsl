// Bilateral filter - edge-preserving smoothing
// Combines spatial and intensity similarity

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    kernel_size: u32,
    sigma_color: f32,    // Range kernel std dev
    sigma_space: f32,    // Spatial kernel std dev
    _pad0: f32,
    _pad1: f32,
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let half_k = i32(params.kernel_size) / 2;
    let center_idx = (y * params.width + x) * params.channels;

    let two_sigma_color_sq = 2.0 * params.sigma_color * params.sigma_color;
    let two_sigma_space_sq = 2.0 * params.sigma_space * params.sigma_space;

    for (var c: u32 = 0u; c < params.channels; c++) {
        let center_val = f32(input[center_idx + c]);
        var sum = 0.0;
        var weight_sum = 0.0;

        for (var ky: i32 = -half_k; ky <= half_k; ky++) {
            for (var kx: i32 = -half_k; kx <= half_k; kx++) {
                let py = clamp(i32(y) + ky, 0, i32(params.height) - 1);
                let px = clamp(i32(x) + kx, 0, i32(params.width) - 1);
                let neighbor_idx = (u32(py) * params.width + u32(px)) * params.channels + c;
                let neighbor_val = f32(input[neighbor_idx]);

                // Spatial weight (Gaussian based on distance)
                let space_dist = f32(kx * kx + ky * ky);
                let space_weight = exp(-space_dist / two_sigma_space_sq);

                // Range weight (Gaussian based on intensity difference)
                let color_dist = (center_val - neighbor_val) * (center_val - neighbor_val);
                let color_weight = exp(-color_dist / two_sigma_color_sq);

                let weight = space_weight * color_weight;
                sum += neighbor_val * weight;
                weight_sum += weight;
            }
        }

        output[center_idx + c] = u32(sum / weight_sum);
    }
}
