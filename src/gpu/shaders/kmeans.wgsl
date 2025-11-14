// GPU kmeans - Auto-generated
// Template: clustering

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

// K-means clustering assigns each pixel to nearest cluster center
// Note: This is the assignment step. Centroid updates require multi-pass approach.

@compute @workgroup_size(256, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_idx = global_id.x;
    let total_pixels = params.width * params.height;

    if (pixel_idx >= total_pixels) {
        return;
    }

    // Read pixel color (assuming 4 channels: RGBA)
    let pixel_base = pixel_idx * params.channels;
    let r = f32(read_byte(&input, pixel_base + 0u));
    let g = f32(read_byte(&input, pixel_base + 1u));
    let b = f32(read_byte(&input, pixel_base + 2u));

    // K-means parameters (hardcoded for now, should be uniforms)
    let k: u32 = 8u;  // Number of clusters

    // Find nearest cluster center
    // Cluster centers should be passed as additional buffer, but for now
    // we'll use a simple approach: divide color space into k regions
    var min_dist: f32 = 1000000.0;
    var nearest_cluster: u32 = 0u;

    for (var cluster: u32 = 0u; cluster < k; cluster++) {
        // Generate cluster centers from color space subdivision
        // This is simplified - proper implementation needs actual centroid buffer
        let step = 256.0 / f32(k);
        let cr = step * (f32(cluster) + 0.5);
        let cg = step * (f32(cluster) + 0.5);
        let cb = step * (f32(cluster) + 0.5);

        // Euclidean distance in RGB space
        let dr = r - cr;
        let dg = g - cg;
        let db = b - cb;
        let dist = dr * dr + dg * dg + db * db;

        if (dist < min_dist) {
            min_dist = dist;
            nearest_cluster = cluster;
        }
    }

    // Assign cluster color (for visualization)
    let step = 256.0 / f32(k);
    let cluster_color = step * (f32(nearest_cluster) + 0.5);

    write_byte(&output, pixel_base + 0u, u32(cluster_color));
    write_byte(&output, pixel_base + 1u, u32(cluster_color));
    write_byte(&output, pixel_base + 2u, u32(cluster_color));
    if (params.channels > 3u) {
        write_byte(&output, pixel_base + 3u, 255u);  // Alpha
    }
}
