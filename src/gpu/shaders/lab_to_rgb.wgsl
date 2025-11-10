// Lab to RGB color space conversion

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
}

// Lab to XYZ conversion
fn lab_to_xyz(lab: vec3<f32>) -> vec3<f32> {
    let l = lab.x;
    let a = lab.y;
    let b = lab.z;

    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;

    let delta = 6.0 / 29.0;
    let delta_cube = delta * delta * delta;

    var x = select(3.0 * delta * delta * (fx - 4.0 / 29.0), pow(fx, 3.0), pow(fx, 3.0) > delta_cube);
    var y = select(3.0 * delta * delta * (fy - 4.0 / 29.0), pow(fy, 3.0), pow(fy, 3.0) > delta_cube);
    var z = select(3.0 * delta * delta * (fz - 4.0 / 29.0), pow(fz, 3.0), pow(fz, 3.0) > delta_cube);

    // D65 white point
    x *= 0.95047;
    y *= 1.00000;
    z *= 1.08883;

    return vec3<f32>(x, y, z);
}

// XYZ to RGB conversion
fn xyz_to_rgb(xyz: vec3<f32>) -> vec3<f32> {
    // Convert from XYZ to linear RGB
    var r = xyz.x *  3.2404542 + xyz.y * -1.5371385 + xyz.z * -0.4985314;
    var g = xyz.x * -0.9692660 + xyz.y *  1.8760108 + xyz.z *  0.0415560;
    var b = xyz.x *  0.0556434 + xyz.y * -0.2040259 + xyz.z *  1.0572252;

    // Apply gamma correction
    r = select(r * 12.92, 1.055 * pow(r, 1.0 / 2.4) - 0.055, r > 0.0031308);
    g = select(g * 12.92, 1.055 * pow(g, 1.0 / 2.4) - 0.055, g > 0.0031308);
    b = select(b * 12.92, 1.055 * pow(b, 1.0 / 2.4) - 0.055, b > 0.0031308);

    return vec3<f32>(r * 255.0, g * 255.0, b * 255.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = (y * params.width + x) * params.channels;

    // Unpack Lab values (L: 0-255 -> 0-100, a/b: 0-255 -> -128..127)
    let l = f32(input[idx]) / 2.55;
    let a = f32(input[idx + 1u]) - 128.0;
    let b = f32(input[idx + 2u]) - 128.0;

    let lab = vec3<f32>(l, a, b);
    let xyz = lab_to_xyz(lab);
    let rgb = xyz_to_rgb(xyz);

    output[idx] = u32(clamp(rgb.x, 0.0, 255.0));
    output[idx + 1u] = u32(clamp(rgb.y, 0.0, 255.0));
    output[idx + 2u] = u32(clamp(rgb.z, 0.0, 255.0));
}
