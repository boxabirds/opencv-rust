// Median blur shader using sorting network for small kernels
// Supports kernel sizes 3, 5

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    kernel_size: u32,
}

// Swap two values if first > second
fn swap(a: ptr<function, u32>, b: ptr<function, u32>) {
    let temp = min(*a, *b);
    *b = max(*a, *b);
    *a = temp;
}

// 3x3 median (9 elements) - optimal sorting network
fn median9(values: array<u32, 9>) -> u32 {
    var v = values;
    swap(&v[0], &v[1]); swap(&v[3], &v[4]); swap(&v[6], &v[7]);
    swap(&v[1], &v[2]); swap(&v[4], &v[5]); swap(&v[7], &v[8]);
    swap(&v[0], &v[1]); swap(&v[3], &v[4]); swap(&v[6], &v[7]);
    swap(&v[0], &v[3]); swap(&v[3], &v[6]);
    swap(&v[0], &v[3]);
    swap(&v[1], &v[4]); swap(&v[4], &v[7]);
    swap(&v[1], &v[4]);
    swap(&v[2], &v[5]); swap(&v[5], &v[8]);
    swap(&v[2], &v[5]);
    swap(&v[1], &v[3]); swap(&v[5], &v[7]);
    swap(&v[2], &v[6]); swap(&v[4], &v[6]);
    swap(&v[2], &v[4]);
    swap(&v[2], &v[3]);
    return v[4]; // Middle element
}

// 5x5 median (25 elements) - partial sorting to find median
fn median25(values: array<u32, 25>) -> u32 {
    var v = values;
    // Partial sort to get median at position 12
    for (var i: u32 = 0u; i < 13u; i++) {
        var min_idx = i;
        for (var j: u32 = i + 1u; j < 25u; j++) {
            if (v[j] < v[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            let temp = v[i];
            v[i] = v[min_idx];
            v[min_idx] = temp;
        }
    }
    return v[12];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let half_k = i32(params.kernel_size) / 2;
    let out_idx = (y * params.width + x) * params.channels;

    for (var c: u32 = 0u; c < params.channels; c++) {
        if (params.kernel_size == 3u) {
            var values: array<u32, 9>;
            var idx: u32 = 0u;

            for (var ky: i32 = -1; ky <= 1; ky++) {
                for (var kx: i32 = -1; kx <= 1; kx++) {
                    let py = clamp(i32(y) + ky, 0, i32(params.height) - 1);
                    let px = clamp(i32(x) + kx, 0, i32(params.width) - 1);
                    let src_idx = (u32(py) * params.width + u32(px)) * params.channels + c;
                    values[idx] = input[src_idx];
                    idx++;
                }
            }
            output[out_idx + c] = median9(values);
        } else if (params.kernel_size == 5u) {
            var values: array<u32, 25>;
            var idx: u32 = 0u;

            for (var ky: i32 = -2; ky <= 2; ky++) {
                for (var kx: i32 = -2; kx <= 2; kx++) {
                    let py = clamp(i32(y) + ky, 0, i32(params.height) - 1);
                    let px = clamp(i32(x) + kx, 0, i32(params.width) - 1);
                    let src_idx = (u32(py) * params.width + u32(px)) * params.channels + c;
                    values[idx] = input[src_idx];
                    idx++;
                }
            }
            output[out_idx + c] = median25(values);
        } else {
            // Fallback: just copy
            let src_idx = (y * params.width + x) * params.channels + c;
            output[out_idx + c] = input[src_idx];
        }
    }
}
