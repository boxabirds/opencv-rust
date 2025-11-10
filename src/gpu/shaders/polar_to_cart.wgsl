@group(0) @binding(0) var<storage, read> input_magnitude: array<f32>;
@group(0) @binding(1) var<storage, read> input_angle: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_x: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    angle_in_degrees: u32,
    _pad: u32,
}

const PI: f32 = 3.14159265359;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.width + x;
    let magnitude = input_magnitude[idx];
    var angle = input_angle[idx];

    if (params.angle_in_degrees != 0u) {
        angle = angle * PI / 180.0;
    }

    output_x[idx] = magnitude * cos(angle);
    output_y[idx] = magnitude * sin(angle);
}
