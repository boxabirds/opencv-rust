// Canny Edge Detection Shader
// Multi-stage GPU-accelerated edge detection

struct CannyParams {
    width: u32,
    height: u32,
    channels: u32,
    low_threshold: u32,
    high_threshold: u32,
    _padding: array<u32, 3>, // Align to 16 bytes
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: CannyParams;

// Sobel kernels for gradient computation
const SOBEL_X: array<i32, 9> = array<i32, 9>(-1, 0, 1, -2, 0, 2, -1, 0, 1);
const SOBEL_Y: array<i32, 9> = array<i32, 9>(-1, -2, -1, 0, 0, 0, 1, 2, 1);

// Get pixel value safely (clamp to edges)
fn get_pixel(x: i32, y: i32) -> u32 {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    let idx = (u32(cx) + u32(cy) * params.width) * params.channels;

    // Convert to grayscale if multi-channel
    if (params.channels == 1u) {
        return input[idx];
    } else if (params.channels == 3u) {
        // RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        let r = f32(input[idx]);
        let g = f32(input[idx + 1u]);
        let b = f32(input[idx + 2u]);
        return u32((0.299 * r + 0.587 * g + 0.114 * b));
    } else if (params.channels == 4u) {
        // RGBA to grayscale
        let r = f32(input[idx]);
        let g = f32(input[idx + 1u]);
        let b = f32(input[idx + 2u]);
        return u32((0.299 * r + 0.587 * g + 0.114 * b));
    }
    return input[idx];
}

// Sobel gradient computation with non-maximum suppression
@compute @workgroup_size(16, 16)
fn canny_edge(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    // Skip borders
    if (x == 0u || y == 0u || x >= params.width - 1u || y >= params.height - 1u) {
        for (var ch = 0u; ch < params.channels; ch++) {
            let idx = (x + y * params.width) * params.channels + ch;
            output[idx] = 0u;
        }
        return;
    }

    // Compute Sobel gradients
    var gx = 0.0;
    var gy = 0.0;

    for (var ky = 0; ky < 3; ky++) {
        for (var kx = 0; kx < 3; kx++) {
            let px = i32(x) + kx - 1;
            let py = i32(y) + ky - 1;
            let pixel = f32(get_pixel(px, py));
            let k_idx = ky * 3 + kx;

            gx += pixel * f32(SOBEL_X[k_idx]);
            gy += pixel * f32(SOBEL_Y[k_idx]);
        }
    }

    // Gradient magnitude
    let magnitude = sqrt(gx * gx + gy * gy);

    // Gradient direction (quantized to 4 directions: 0°, 45°, 90°, 135°)
    let angle = atan2(gy, gx);

    // Non-maximum suppression
    var is_max = true;
    let angle_deg = degrees(angle);

    // Normalize angle to [0, 180)
    var normalized_angle = angle_deg;
    if (normalized_angle < 0.0) {
        normalized_angle += 180.0;
    }

    // Check neighbors in gradient direction
    var n1_mag = 0.0;
    var n2_mag = 0.0;

    if ((normalized_angle >= 0.0 && normalized_angle < 22.5) ||
        (normalized_angle >= 157.5 && normalized_angle < 180.0)) {
        // Horizontal edge (0°)
        let p1 = get_pixel(i32(x) + 1, i32(y));
        let p2 = get_pixel(i32(x) - 1, i32(y));
        n1_mag = f32(p1);
        n2_mag = f32(p2);
    } else if (normalized_angle >= 22.5 && normalized_angle < 67.5) {
        // Diagonal edge (45°)
        let p1 = get_pixel(i32(x) + 1, i32(y) - 1);
        let p2 = get_pixel(i32(x) - 1, i32(y) + 1);
        n1_mag = f32(p1);
        n2_mag = f32(p2);
    } else if (normalized_angle >= 67.5 && normalized_angle < 112.5) {
        // Vertical edge (90°)
        let p1 = get_pixel(i32(x), i32(y) + 1);
        let p2 = get_pixel(i32(x), i32(y) - 1);
        n1_mag = f32(p1);
        n2_mag = f32(p2);
    } else {
        // Diagonal edge (135°)
        let p1 = get_pixel(i32(x) - 1, i32(y) - 1);
        let p2 = get_pixel(i32(x) + 1, i32(y) + 1);
        n1_mag = f32(p1);
        n2_mag = f32(p2);
    }

    // Suppress if not a local maximum
    if (magnitude <= n1_mag || magnitude <= n2_mag) {
        is_max = false;
    }

    // Double threshold
    var edge_value = 0u;
    if (is_max) {
        let mag_u32 = u32(clamp(magnitude, 0.0, 255.0));
        if (mag_u32 >= params.high_threshold) {
            edge_value = 255u; // Strong edge
        } else if (mag_u32 >= params.low_threshold) {
            edge_value = 128u; // Weak edge (simplified - in full Canny, this would be connected)
        }
    }

    // Write output (same value for all channels)
    for (var ch = 0u; ch < params.channels; ch++) {
        let out_idx = (x + y * params.width) * params.channels + ch;
        output[out_idx] = edge_value;
    }
}
