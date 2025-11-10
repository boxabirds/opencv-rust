// Rotate image by 90, 180, or 270 degrees shader

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    rotate_code: i32,  // 0=90cw, 1=180, 2=90ccw
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    let out_width = select(params.width, params.height, params.rotate_code != 1);
    let out_height = select(params.height, params.width, params.rotate_code != 1);

    if (x >= out_width || y >= out_height) {
        return;
    }

    var src_x: u32;
    var src_y: u32;

    if (params.rotate_code == 0) {
        // 90 clockwise: (x,y) -> (h-1-y, x)
        src_x = y;
        src_y = params.height - 1u - x;
    } else if (params.rotate_code == 1) {
        // 180 degrees: (x,y) -> (w-1-x, h-1-y)
        src_x = params.width - 1u - x;
        src_y = params.height - 1u - y;
    } else {
        // 90 counter-clockwise: (x,y) -> (y, w-1-x)
        src_x = params.width - 1u - y;
        src_y = x;
    }

    let src_idx = (src_y * params.width + src_x) * params.channels;
    let dst_idx = (y * out_width + x) * params.channels;

    // Copy all channels
    for (var c: u32 = 0u; c < params.channels; c++) {
        output[dst_idx + c] = input[src_idx + c];
    }
}
