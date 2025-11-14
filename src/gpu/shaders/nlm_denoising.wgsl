// GPU nlm_denoising - Auto-generated
// Template: filter_parallel

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
}

// Byte access helpers
fn read_byte(buffer: ptr<storage, array<u32>, read>, byte_index: u32) -> u32 {
    let u32_index = byte_index / 4u;
    let byte_offset = byte_index % 4u;
    let word = buffer[u32_index];
    return (word >> (byte_offset * 8u)) & 0xFFu;
}

fn write_byte(buffer: ptr<storage, array<u32>, read_write>, byte_index: u32, value: u32) {
    let u32_index = byte_index / 4u;
    let byte_offset = byte_index % 4u;
    let old_word = buffer[u32_index];
    let mask = ~(0xFFu << (byte_offset * 8u));
    let new_word = (old_word & mask) | ((value & 0xFFu) << (byte_offset * 8u));
    buffer[u32_index] = new_word;
}

// Fast exp approximation for GPU (avoids slow native exp)
fn fast_exp(x: f32) -> f32 {
    let a = 1.0 + x / 256.0;
    let a2 = a * a;
    let a4 = a2 * a2;
    let a8 = a4 * a4;
    let a16 = a8 * a8;
    let a32 = a16 * a16;
    let a64 = a32 * a32;
    let a128 = a64 * a64;
    let a256 = a128 * a128;
    return a256;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.width + x;
    let pixel_base = idx * params.channels;

    // Non-Local Means parameters
    let search_window: i32 = 10;  // Search radius
    let patch_size: i32 = 3;       // Patch radius
    let h: f32 = 10.0;             // Filtering strength
    let h2 = h * h;

    // Process each channel
    for (var c: u32 = 0u; c < params.channels; c++) {
        var sum_weights: f32 = 0.0;
        var sum_pixels: f32 = 0.0;

        // Search in neighborhood
        for (var dy: i32 = -search_window; dy <= search_window; dy++) {
            for (var dx: i32 = -search_window; dx <= search_window; dx++) {
                let nx = i32(x) + dx;
                let ny = i32(y) + dy;

                if (nx < 0 || nx >= i32(params.width) || ny < 0 || ny >= i32(params.height)) {
                    continue;
                }

                // Compute patch distance
                var patch_dist: f32 = 0.0;
                var patch_count: f32 = 0.0;

                for (var py: i32 = -patch_size; py <= patch_size; py++) {
                    for (var px: i32 = -patch_size; px <= patch_size; px++) {
                        let cx1 = i32(x) + px;
                        let cy1 = i32(y) + py;
                        let cx2 = nx + px;
                        let cy2 = ny + py;

                        if (cx1 >= 0 && cx1 < i32(params.width) && cy1 >= 0 && cy1 < i32(params.height) &&
                            cx2 >= 0 && cx2 < i32(params.width) && cy2 >= 0 && cy2 < i32(params.height)) {

                            let idx1 = u32(cy1) * params.width + u32(cx1);
                            let idx2 = u32(cy2) * params.width + u32(cx2);
                            let pb1 = idx1 * params.channels + c;
                            let pb2 = idx2 * params.channels + c;

                            let v1 = f32(read_byte(&input, pb1));
                            let v2 = f32(read_byte(&input, pb2));
                            let diff = v1 - v2;
                            patch_dist += diff * diff;
                            patch_count += 1.0;
                        }
                    }
                }

                // Normalize patch distance
                if (patch_count > 0.0) {
                    patch_dist = patch_dist / patch_count;
                }

                // Compute weight using Gaussian
                let weight = fast_exp(-patch_dist / h2);

                let nidx = u32(ny) * params.width + u32(nx);
                let npb = nidx * params.channels + c;
                let pixel_val = f32(read_byte(&input, npb));

                sum_weights += weight;
                sum_pixels += weight * pixel_val;
            }
        }

        // Normalize and write result
        let result = sum_pixels / max(sum_weights, 0.0001);
        write_byte(&output, pixel_base + c, u32(clamp(result, 0.0, 255.0)));
    }
}
