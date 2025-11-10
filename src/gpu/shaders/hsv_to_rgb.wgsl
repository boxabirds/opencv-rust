// HSV to RGB color space conversion

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

    // HSV input: H in [0, 180], S in [0, 255], V in [0, 255] (OpenCV format)
    let h = f32(input[idx]) * 2.0;  // Convert back to [0, 360]
    let s = f32(input[idx + 1u]) / 255.0;
    let v = f32(input[idx + 2u]) / 255.0;

    let c = v * s;
    let h_prime = h / 60.0;
    let x_val = c * (1.0 - abs(h_prime % 2.0 - 1.0));
    let m = v - c;

    var r: f32;
    var g: f32;
    var b: f32;

    if (h_prime < 1.0) {
        r = c; g = x_val; b = 0.0;
    } else if (h_prime < 2.0) {
        r = x_val; g = c; b = 0.0;
    } else if (h_prime < 3.0) {
        r = 0.0; g = c; b = x_val;
    } else if (h_prime < 4.0) {
        r = 0.0; g = x_val; b = c;
    } else if (h_prime < 5.0) {
        r = x_val; g = 0.0; b = c;
    } else {
        r = c; g = 0.0; b = x_val;
    }

    output[idx] = u32((r + m) * 255.0);
    output[idx + 1u] = u32((g + m) * 255.0);
    output[idx + 2u] = u32((b + m) * 255.0);
}
