// Bilinear Resize Shader
// GPU-accelerated image resizing with bilinear interpolation

struct ResizeParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
    _padding: array<u32, 3>, // Align to 16 bytes
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: ResizeParams;

// Bilinear interpolation
fn bilinear_sample(x: f32, y: f32, ch: u32) -> f32 {
    // Get integer coordinates
    let x0 = u32(floor(x));
    let y0 = u32(floor(y));
    let x1 = min(x0 + 1u, params.src_width - 1u);
    let y1 = min(y0 + 1u, params.src_height - 1u);

    // Get fractional parts
    let fx = fract(x);
    let fy = fract(y);

    // Sample four pixels
    let idx00 = (x0 + y0 * params.src_width) * params.channels + ch;
    let idx10 = (x1 + y0 * params.src_width) * params.channels + ch;
    let idx01 = (x0 + y1 * params.src_width) * params.channels + ch;
    let idx11 = (x1 + y1 * params.src_width) * params.channels + ch;

    let v00 = f32(input[idx00]);
    let v10 = f32(input[idx10]);
    let v01 = f32(input[idx01]);
    let v11 = f32(input[idx11]);

    // Bilinear interpolation
    let v0 = mix(v00, v10, fx);
    let v1 = mix(v01, v11, fx);
    return mix(v0, v1, fy);
}

@compute @workgroup_size(16, 16)
fn resize_bilinear(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst_x = id.x;
    let dst_y = id.y;

    if (dst_x >= params.dst_width || dst_y >= params.dst_height) {
        return;
    }

    // Calculate source coordinates
    let scale_x = f32(params.src_width) / f32(params.dst_width);
    let scale_y = f32(params.src_height) / f32(params.dst_height);

    let src_x = (f32(dst_x) + 0.5) * scale_x - 0.5;
    let src_y = (f32(dst_y) + 0.5) * scale_y - 0.5;

    // Clamp to valid range
    let clamped_x = clamp(src_x, 0.0, f32(params.src_width - 1u));
    let clamped_y = clamp(src_y, 0.0, f32(params.src_height - 1u));

    // Process each channel
    for (var ch = 0u; ch < params.channels; ch++) {
        let value = bilinear_sample(clamped_x, clamped_y, ch);
        let out_idx = (dst_x + dst_y * params.dst_width) * params.channels + ch;
        output[out_idx] = u32(clamp(value, 0.0, 255.0));
    }
}
