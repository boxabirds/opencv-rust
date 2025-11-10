// RGB to Grayscale conversion shader
// Uses standard luminance formula: 0.299*R + 0.587*G + 0.114*B

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,  // Should be 3 for RGB
    _pad: u32,
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx_color = (y * params.width + x) * params.channels;
    let idx_gray = y * params.width + x;

    // Read RGB values
    let r = f32(input[idx_color]);
    let g = f32(input[idx_color + 1u]);
    let b = f32(input[idx_color + 2u]);

    // Convert to grayscale using luminance formula
    let gray = 0.299 * r + 0.587 * g + 0.114 * b;

    output[idx_gray] = u32(clamp(gray, 0.0, 255.0));
}
