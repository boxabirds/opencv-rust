// Adaptive threshold shader using mean method

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    max_value: u32,
    block_size: u32,  // Must be odd
    c_constant: i32,   // Constant subtracted from mean
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let half_block = i32(params.block_size) / 2;
    var sum: f32 = 0.0;
    var count: u32 = 0u;

    // Compute local mean
    for (var ky: i32 = -half_block; ky <= half_block; ky++) {
        for (var kx: i32 = -half_block; kx <= half_block; kx++) {
            let py = clamp(i32(y) + ky, 0, i32(params.height) - 1);
            let px = clamp(i32(x) + kx, 0, i32(params.width) - 1);
            let idx = u32(py) * params.width + u32(px);
            sum += f32(input[idx]);
            count++;
        }
    }

    let mean = sum / f32(count);
    let threshold = mean - f32(params.c_constant);

    let pixel_value = f32(input[y * params.width + x]);

    // Apply threshold
    let result = select(0u, params.max_value, pixel_value > threshold);
    output[y * params.width + x] = result;
}
