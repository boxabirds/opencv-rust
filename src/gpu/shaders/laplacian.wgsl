// Laplacian edge detection shader

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

// 3x3 Laplacian kernel
const KERNEL: array<array<f32, 3>, 3> = array(
    array( 0.0,  1.0,  0.0),
    array( 1.0, -4.0,  1.0),
    array( 0.0,  1.0,  0.0),
);

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    // Skip border pixels
    if (x == 0u || y == 0u || x >= params.width - 1u || y >= params.height - 1u) {
        output[y * params.width + x] = 0u;
        return;
    }

    var sum: f32 = 0.0;

    // Apply Laplacian kernel
    for (var ky: i32 = -1; ky <= 1; ky++) {
        for (var kx: i32 = -1; kx <= 1; kx++) {
            let py = i32(y) + ky;
            let px = i32(x) + kx;
            let idx = u32(py) * params.width + u32(px);
            let pixel_value = f32(input[idx]);
            let kernel_value = KERNEL[ky + 1][kx + 1];
            sum += pixel_value * kernel_value;
        }
    }

    // Store absolute value
    output[y * params.width + x] = u32(clamp(abs(sum), 0.0, 255.0));
}
