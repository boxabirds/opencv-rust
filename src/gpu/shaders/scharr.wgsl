// Scharr edge detection shader (more accurate than Sobel)

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    dx: u32,  // Compute X gradient if > 0
    dy: u32,  // Compute Y gradient if > 0
}

// Scharr kernels (higher accuracy than Sobel)
const SCHARR_X: array<array<f32, 3>, 3> = array(
    array( -3.0,  0.0,  3.0),
    array(-10.0,  0.0, 10.0),
    array( -3.0,  0.0,  3.0),
);

const SCHARR_Y: array<array<f32, 3>, 3> = array(
    array( -3.0, -10.0, -3.0),
    array(  0.0,   0.0,  0.0),
    array(  3.0,  10.0,  3.0),
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

    // Apply Scharr X kernel if requested
    if (params.dx > 0u) {
        for (var ky: i32 = -1; ky <= 1; ky++) {
            for (var kx: i32 = -1; kx <= 1; kx++) {
                let py = i32(y) + ky;
                let px = i32(x) + kx;
                let idx = u32(py) * params.width + u32(px);
                sum += f32(input[idx]) * SCHARR_X[ky + 1][kx + 1];
            }
        }
    }

    // Apply Scharr Y kernel if requested
    if (params.dy > 0u) {
        for (var ky: i32 = -1; ky <= 1; ky++) {
            for (var kx: i32 = -1; kx <= 1; kx++) {
                let py = i32(y) + ky;
                let px = i32(x) + kx;
                let idx = u32(py) * params.width + u32(px);
                sum += f32(input[idx]) * SCHARR_Y[ky + 1][kx + 1];
            }
        }
    }

    output[y * params.width + x] = u32(clamp(abs(sum), 0.0, 255.0));
}
