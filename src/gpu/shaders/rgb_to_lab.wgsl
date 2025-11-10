// RGB to Lab color space conversion
// Lab is perceptually uniform color space

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
}

// RGB to XYZ conversion
fn rgb_to_xyz(r: f32, g: f32, b: f32) -> vec3<f32> {
    // Normalize and apply gamma correction
    var rr = r / 255.0;
    var gg = g / 255.0;
    var bb = b / 255.0;

    rr = select(rr / 12.92, pow((rr + 0.055) / 1.055, 2.4), rr > 0.04045);
    gg = select(gg / 12.92, pow((gg + 0.055) / 1.055, 2.4), gg > 0.04045);
    bb = select(bb / 12.92, pow((bb + 0.055) / 1.055, 2.4), bb > 0.04045);

    // Convert to XYZ (D65 illuminant)
    let x = rr * 0.4124564 + gg * 0.3575761 + bb * 0.1804375;
    let y = rr * 0.2126729 + gg * 0.7151522 + bb * 0.0721750;
    let z = rr * 0.0193339 + gg * 0.1191920 + bb * 0.9503041;

    return vec3<f32>(x, y, z);
}

// XYZ to Lab conversion
fn xyz_to_lab(xyz: vec3<f32>) -> vec3<f32> {
    // D65 white point
    let xn = 0.95047;
    let yn = 1.00000;
    let zn = 1.08883;

    var fx = xyz.x / xn;
    var fy = xyz.y / yn;
    var fz = xyz.z / zn;

    let delta = 6.0 / 29.0;
    let delta_sq = delta * delta;
    let delta_cube = delta_sq * delta;

    fx = select(pow(fx, 1.0 / 3.0), (fx / (3.0 * delta_sq)) + (4.0 / 29.0), fx > delta_cube);
    fy = select(pow(fy, 1.0 / 3.0), (fy / (3.0 / delta_sq)) + (4.0 / 29.0), fy > delta_cube);
    fz = select(pow(fz, 1.0 / 3.0), (fz / (3.0 * delta_sq)) + (4.0 / 29.0), fz > delta_cube);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);

    return vec3<f32>(l, a, b);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = (y * params.width + x) * params.channels;

    let r = f32(input[idx]);
    let g = f32(input[idx + 1u]);
    let b = f32(input[idx + 2u]);

    let xyz = rgb_to_xyz(r, g, b);
    let lab = xyz_to_lab(xyz);

    // Store Lab values (L in [0, 100], a and b in [-128, 127] shifted to [0, 255])
    output[idx] = u32(clamp(lab.x * 2.55, 0.0, 255.0));  // L: 0-100 -> 0-255
    output[idx + 1u] = u32(clamp(lab.y + 128.0, 0.0, 255.0));  // a: -128..127 -> 0..255
    output[idx + 2u] = u32(clamp(lab.z + 128.0, 0.0, 255.0));  // b: -128..127 -> 0..255
}
