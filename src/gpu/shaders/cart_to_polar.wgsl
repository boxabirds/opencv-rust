@group(0) @binding(0) var<storage, read> input_x: array<f32>;
@group(0) @binding(1) var<storage, read> input_y: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_magnitude: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_angle: array<f32>;
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
    let x_val = input_x[idx];
    let y_val = input_y[idx];

    output_magnitude[idx] = sqrt(x_val * x_val + y_val * y_val);

    let angle = atan2(y_val, x_val);
    if (params.angle_in_degrees != 0u) {
        output_angle[idx] = angle * 180.0 / PI;
    } else {
        output_angle[idx] = angle;
    }
}
