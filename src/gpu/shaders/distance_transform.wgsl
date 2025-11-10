// Distance transform - compute distance to nearest zero pixel
// Uses simple Euclidean distance approximation

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    max_dist: f32,
    _pad: u32,
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.width + x;

    // If pixel is zero, distance is 0
    if (input[idx] == 0u) {
        output[idx] = 0.0;
        return;
    }

    // Find minimum distance to any zero pixel in neighborhood
    var min_dist = params.max_dist;

    let search_radius = 10i;  // Search within 10 pixel radius
    for (var dy: i32 = -search_radius; dy <= search_radius; dy++) {
        for (var dx: i32 = -search_radius; dx <= search_radius; dx++) {
            let py = clamp(i32(y) + dy, 0, i32(params.height) - 1);
            let px = clamp(i32(x) + dx, 0, i32(params.width) - 1);
            let neighbor_idx = u32(py) * params.width + u32(px);

            if (input[neighbor_idx] == 0u) {
                let dist = sqrt(f32(dx * dx + dy * dy));
                min_dist = min(min_dist, dist);
            }
        }
    }

    output[idx] = min_dist;
}
