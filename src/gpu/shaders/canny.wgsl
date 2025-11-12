// Canny Edge Detection Shader
// Multi-stage GPU-accelerated edge detection

struct CannyParams {
    width: u32,
    height: u32,
    channels: u32,
    low_threshold: u32,
    high_threshold: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: CannyParams;

// Sobel kernels for gradient computation
const SOBEL_X: array<i32, 9> = array<i32, 9>(-1, 0, 1, -2, 0, 2, -1, 0, 1);
const SOBEL_Y: array<i32, 9> = array<i32, 9>(-1, -2, -1, 0, 0, 0, 1, 2, 1);

// Get pixel value safely (clamp to edges)

// === Byte Access Helpers ===
// Required for correct RGBA byte extraction from u32 storage buffers

// Byte Access Helpers for GPU Shaders
//
// WebGPU storage buffers must be 4-byte aligned, so byte arrays are stored as array<u32>.
// These helpers extract and write individual bytes from the packed u32 values.
//
// Memory layout (little-endian):
// - Bytes [R, G, B, A] at indices [0, 1, 2, 3]
// - Stored in u32[0] as: A << 24 | B << 16 | G << 8 | R
//
// Usage:
//   let r_value = read_byte(&input, pixel_index * 4u + 0u);  // Read R channel
//   write_byte(&output, pixel_index * 4u + 0u, r_value);     // Write R channel

/// Read a single byte from a u32 storage buffer
///
/// @param buffer - Pointer to u32 storage buffer
/// @param byte_index - Byte index (0, 1, 2, 3, 4, 5, ...)
/// @return Byte value (0-255)
fn read_byte(buffer: ptr<storage, array<u32>, read>, byte_index: u32) -> u32 {
    let u32_index = byte_index / 4u;           // Which u32 word
    let byte_offset = byte_index % 4u;         // Which byte in that word (0-3)
    let word = buffer[u32_index];              // Read the full u32
    return (word >> (byte_offset * 8u)) & 0xFFu; // Extract the byte
}

/// Read a single byte from a read-write u32 storage buffer
fn read_byte_rw(buffer: ptr<storage, array<u32>, read_write>, byte_index: u32) -> u32 {
    let u32_index = byte_index / 4u;
    let byte_offset = byte_index % 4u;
    let word = buffer[u32_index];
    return (word >> (byte_offset * 8u)) & 0xFFu;
}

/// Write a single byte to a u32 storage buffer
///
/// @param buffer - Pointer to u32 storage buffer (read_write)
/// @param byte_index - Byte index (0, 1, 2, 3, 4, 5, ...)
/// @param value - Byte value to write (0-255)
fn write_byte(buffer: ptr<storage, array<u32>, read_write>, byte_index: u32, value: u32) {
    let u32_index = byte_index / 4u;
    let byte_offset = byte_index % 4u;

    // Read-modify-write the u32 word
    let old_word = buffer[u32_index];
    let mask = ~(0xFFu << (byte_offset * 8u));  // Mask to clear the target byte
    let new_word = (old_word & mask) | ((value & 0xFFu) << (byte_offset * 8u));
    buffer[u32_index] = new_word;
}

/// Read 4 bytes (RGBA pixel) from buffer
///
/// @param buffer - Pointer to u32 storage buffer
/// @param pixel_index - Pixel index (not byte index!)
/// @param channels - Number of channels (3 for RGB, 4 for RGBA)
/// @return vec4<u32> with R, G, B, A channels (A=255 if channels==3)
fn read_pixel(buffer: ptr<storage, array<u32>, read>, pixel_index: u32, channels: u32) -> vec4<u32> {
    let base = pixel_index * channels;

    if (channels >= 4u) {
        return vec4<u32>(
            read_byte(buffer, base + 0u),  // R
            read_byte(buffer, base + 1u),  // G
            read_byte(buffer, base + 2u),  // B
            read_byte(buffer, base + 3u)   // A
        );
    } else {
        return vec4<u32>(
            read_byte(buffer, base + 0u),  // R
            read_byte(buffer, base + 1u),  // G
            read_byte(buffer, base + 2u),  // B
            255u                            // A (default opaque)
        );
    }
}

/// Write 4 bytes (RGBA pixel) to buffer
///
/// @param buffer - Pointer to u32 storage buffer
/// @param pixel_index - Pixel index (not byte index!)
/// @param pixel - vec4<u32> with R, G, B, A channels
/// @param channels - Number of channels to write (3 for RGB, 4 for RGBA)
fn write_pixel(buffer: ptr<storage, array<u32>, read_write>, pixel_index: u32, pixel: vec4<u32>, channels: u32) {
    let base = pixel_index * channels;

    write_byte(buffer, base + 0u, pixel.r);  // R
    write_byte(buffer, base + 1u, pixel.g);  // G
    write_byte(buffer, base + 2u, pixel.b);  // B

    if (channels >= 4u) {
        write_byte(buffer, base + 3u, pixel.a);  // A
    }
}

// === End Byte Access Helpers ===

fn get_pixel(x: i32, y: i32) -> u32 {
    let cx = clamp(x, 0, i32(params.width) - 1);
    let cy = clamp(y, 0, i32(params.height) - 1);
    let idx = (u32(cx) + u32(cy) * params.width) * params.channels;

    // Convert to grayscale if multi-channel
    if (params.channels == 1u) {
        return read_byte(&input, idx);
    } else if (params.channels == 3u) {
        // RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        let r = f32(read_byte(&input, idx));
        let g = f32(read_byte(&input, idx + 1u));
        let b = f32(read_byte(&input, idx + 2u));
        return u32((0.299 * r + 0.587 * g + 0.114 * b));
    } else if (params.channels == 4u) {
        // RGBA to grayscale
        let r = f32(read_byte(&input, idx));
        let g = f32(read_byte(&input, idx + 1u));
        let b = f32(read_byte(&input, idx + 2u));
        return u32((0.299 * r + 0.587 * g + 0.114 * b));
    }
    return read_byte(&input, idx);
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
            write_byte(&output, idx, 0u);
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
        write_byte(&output, out_idx, edge_value);
    }
}
