// Integral image (summed area table) computation
// Uses parallel prefix sum per row, then per column

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    pass: u32,  // 0 = horizontal scan, 1 = vertical scan
    _pad: u32,
}

// Horizontal pass: compute cumulative sum along rows
@compute @workgroup_size(256, 1, 1)
fn horizontal_scan(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = global_id.y;
    if (row >= params.height) {
        return;
    }

    let tid = local_id.x;
    let row_start = row * params.width;

    // Load data
    var sum: u32 = 0u;
    for (var i: u32 = tid; i < params.width; i += 256u) {
        sum += input[row_start + i];
        output[row_start + i] = sum;
    }
}

// Vertical pass: compute cumulative sum along columns
@compute @workgroup_size(16, 16)
fn vertical_scan(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    if (x >= params.width) {
        return;
    }

    var sum: u32 = 0u;
    for (var y: u32 = 0u; y < params.height; y++) {
        let idx = y * params.width + x;
        sum += output[idx];
        output[idx] = sum;
    }
}
