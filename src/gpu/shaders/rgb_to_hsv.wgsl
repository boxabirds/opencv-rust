// RGB to HSV conversion shader

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

    // Read RGB values (normalized to [0,1])
    let r = f32(input[idx]) / 255.0;
    let g = f32(input[idx + 1u]) / 255.0;
    let b = f32(input[idx + 2u]) / 255.0;

    // Convert RGB to HSV
    let max_val = max(max(r, g), b);
    let min_val = min(min(r, g), b);
    let delta = max_val - min_val;

    // Hue calculation
    var h: f32 = 0.0;
    if (delta > 0.0) {
        if (max_val == r) {
            h = 60.0 * (((g - b) / delta) % 6.0);
        } else if (max_val == g) {
            h = 60.0 * (((b - r) / delta) + 2.0);
        } else {
            h = 60.0 * (((r - g) / delta) + 4.0);
        }
        if (h < 0.0) {
            h += 360.0;
        }
    }

    // Saturation calculation
    var s: f32 = 0.0;
    if (max_val > 0.0) {
        s = delta / max_val;
    }

    // Value is max_val
    let v = max_val;

    // Store as H(0-180), S(0-255), V(0-255) for OpenCV compatibility
    output[idx] = u32(h / 2.0);  // H in [0, 180]
    output[idx + 1u] = u32(s * 255.0);  // S in [0, 255]
    output[idx + 2u] = u32(v * 255.0);  // V in [0, 255]
}
