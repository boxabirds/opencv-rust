// Farneback Optical Flow - GPU Implementation
// Simplified dense optical flow using block matching

@group(0) @binding(0) var<storage, read> prev_frame: array<u32>;
@group(0) @binding(1) var<storage, read> next_frame: array<u32>;
@group(0) @binding(2) var<storage, read_write> flow: array<f32>;  // 2 channels (dx, dy)
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    block_size: u32,
    search_range: u32,
}

const MAX_BLOCK_SIZE: u32 = 15u;
const MAX_SEARCH_RANGE: u32 = 10u;

// Byte access helpers
fn read_byte(buffer: ptr<storage, array<u32>, read>, byte_index: u32) -> u32 {
    let u32_index = byte_index / 4u;
    let byte_offset = byte_index % 4u;
    let word = buffer[u32_index];
    return (word >> (byte_offset * 8u)) & 0xFFu;
}

// Compute SSD between two blocks
fn compute_ssd(
    prev: ptr<storage, array<u32>, read>,
    next: ptr<storage, array<u32>, read>,
    x1: u32, y1: u32,
    x2: i32, y2: i32,
    block_size: u32,
    width: u32, height: u32
) -> f32 {
    var ssd: f32 = 0.0;
    let half_block = i32(block_size / 2u);

    for (var dy: i32 = -half_block; dy <= half_block; dy++) {
        for (var dx: i32 = -half_block; dx <= half_block; dx++) {
            let py1 = clamp(i32(y1) + dy, 0, i32(height) - 1);
            let px1 = clamp(i32(x1) + dx, 0, i32(width) - 1);
            let py2 = clamp(y2 + dy, 0, i32(height) - 1);
            let px2 = clamp(x2 + dx, 0, i32(width) - 1);

            let idx1 = u32(py1) * width + u32(px1);
            let idx2 = u32(py2) * width + u32(px2);

            let val1 = f32(read_byte(prev, idx1));
            let val2 = f32(read_byte(next, idx2));

            let diff = val1 - val2;
            ssd += diff * diff;
        }
    }

    return ssd;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let half_block = params.block_size / 2u;

    // Skip border pixels
    if (x < half_block || x >= params.width - half_block ||
        y < half_block || y >= params.height - half_block) {
        let flow_idx = (y * params.width + x) * 2u;
        flow[flow_idx] = 0.0;
        flow[flow_idx + 1u] = 0.0;
        return;
    }

    var best_dx: i32 = 0;
    var best_dy: i32 = 0;
    var best_ssd: f32 = 1e10;

    let search = i32(params.search_range);

    // Search for best match in next frame
    for (var dy: i32 = -search; dy <= search; dy++) {
        for (var dx: i32 = -search; dx <= search; dx++) {
            let nx = i32(x) + dx;
            let ny = i32(y) + dy;

            // Check bounds
            if (nx < i32(half_block) || nx >= i32(params.width - half_block) ||
                ny < i32(half_block) || ny >= i32(params.height - half_block)) {
                continue;
            }

            let ssd = compute_ssd(
                &prev_frame, &next_frame,
                x, y, nx, ny,
                params.block_size,
                params.width, params.height
            );

            // Add distance penalty to prefer closer matches
            let dist_penalty = sqrt(f32(dx * dx + dy * dy)) * 0.1;
            let cost = ssd + dist_penalty;

            if (cost < best_ssd) {
                best_ssd = cost;
                best_dx = dx;
                best_dy = dy;
            }
        }
    }

    // Write flow vector
    let flow_idx = (y * params.width + x) * 2u;
    flow[flow_idx] = f32(best_dx);
    flow[flow_idx + 1u] = f32(best_dy);
}
