// Flip image shader (horizontal, vertical, or both)

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    flip_code: i32,  // 0=vertical, 1=horizontal, -1=both
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    var src_x = x;
    var src_y = y;

    // Apply flip transformation
    if (params.flip_code > 0) {
        // Horizontal flip
        src_x = params.width - 1u - x;
    } else if (params.flip_code == 0) {
        // Vertical flip
        src_y = params.height - 1u - y;
    } else {
        // Both
        src_x = params.width - 1u - x;
        src_y = params.height - 1u - y;
    }

    let src_idx = (src_y * params.width + src_x) * params.channels;
    let dst_idx = (y * params.width + x) * params.channels;

    // Copy all channels
    for (var c: u32 = 0u; c < params.channels; c++) {
        output[dst_idx + c] = input[src_idx + c];
    }
}
