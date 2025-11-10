@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> kernel: array<f32>;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    kernel_width: u32,
    kernel_height: u32,
    anchor_x: i32,
    anchor_y: i32,
    _pad: u32,
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let half_kw = i32(params.kernel_width) / 2;
    let half_kh = i32(params.kernel_height) / 2;

    for (var c = 0u; c < params.channels; c++) {
        var sum = 0.0;

        for (var ky = 0i; ky < i32(params.kernel_height); ky++) {
            for (var kx = 0i; kx < i32(params.kernel_width); kx++) {
                let src_x = i32(x) + kx - params.anchor_x;
                let src_y = i32(y) + ky - params.anchor_y;

                // Border handling: clamp to edge
                let clamped_x = clamp(src_x, 0, i32(params.width) - 1);
                let clamped_y = clamp(src_y, 0, i32(params.height) - 1);

                let idx = (u32(clamped_y) * params.width + u32(clamped_x)) * params.channels + c;
                let kernel_idx = u32(ky) * params.kernel_width + u32(kx);

                sum += f32(input[idx]) * kernel[kernel_idx];
            }
        }

        let out_idx = (y * params.width + x) * params.channels + c;
        output[out_idx] = u32(clamp(sum, 0.0, 255.0));
    }
}
