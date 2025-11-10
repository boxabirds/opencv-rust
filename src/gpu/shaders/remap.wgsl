@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> map_x: array<f32>;
@group(0) @binding(2) var<storage, read> map_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
}

fn bilinear_sample(x: f32, y: f32, c: u32) -> f32 {
    let x0 = floor(x);
    let y0 = floor(y);
    let x1 = x0 + 1.0;
    let y1 = y0 + 1.0;

    let fx = x - x0;
    let fy = y - y0;

    let ix0 = i32(x0);
    let iy0 = i32(y0);
    let ix1 = i32(x1);
    let iy1 = i32(y1);

    // Clamp to image boundaries
    let cx0 = clamp(ix0, 0, i32(params.width) - 1);
    let cy0 = clamp(iy0, 0, i32(params.height) - 1);
    let cx1 = clamp(ix1, 0, i32(params.width) - 1);
    let cy1 = clamp(iy1, 0, i32(params.height) - 1);

    let idx00 = (u32(cy0) * params.width + u32(cx0)) * params.channels + c;
    let idx10 = (u32(cy0) * params.width + u32(cx1)) * params.channels + c;
    let idx01 = (u32(cy1) * params.width + u32(cx0)) * params.channels + c;
    let idx11 = (u32(cy1) * params.width + u32(cx1)) * params.channels + c;

    let v00 = f32(input[idx00]);
    let v10 = f32(input[idx10]);
    let v01 = f32(input[idx01]);
    let v11 = f32(input[idx11]);

    let v0 = v00 * (1.0 - fx) + v10 * fx;
    let v1 = v01 * (1.0 - fx) + v11 * fx;

    return v0 * (1.0 - fy) + v1 * fy;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let map_idx = y * params.width + x;
    let src_x = map_x[map_idx];
    let src_y = map_y[map_idx];

    // Check if mapping is valid
    if (src_x < 0.0 || src_x >= f32(params.width) - 1.0 ||
        src_y < 0.0 || src_y >= f32(params.height) - 1.0) {
        // Out of bounds - set to black
        for (var c = 0u; c < params.channels; c++) {
            let idx = (y * params.width + x) * params.channels + c;
            output[idx] = 0u;
        }
        return;
    }

    for (var c = 0u; c < params.channels; c++) {
        let value = bilinear_sample(src_x, src_y, c);
        let idx = (y * params.width + x) * params.channels + c;
        output[idx] = u32(clamp(value, 0.0, 255.0));
    }
}
