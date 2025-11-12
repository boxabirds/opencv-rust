// RGB to Lab color space conversion
// Lab is perceptually uniform color space

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
}

// RGB to XYZ conversion

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

fn rgb_to_xyz(r: f32, g: f32, b: f32) -> vec3<f32> {
    // Normalize and apply gamma correction
    var rr = r / 255.0;
    var gg = g / 255.0;
    var bb = b / 255.0;

    rr = select(rr / 12.92, pow((rr + 0.055) / 1.055, 2.4), rr > 0.04045);
    gg = select(gg / 12.92, pow((gg + 0.055) / 1.055, 2.4), gg > 0.04045);
    bb = select(bb / 12.92, pow((bb + 0.055) / 1.055, 2.4), bb > 0.04045);

    // Convert to XYZ (D65 illuminant)
    let x = rr * 0.4124564 + gg * 0.3575761 + bb * 0.1804375;
    let y = rr * 0.2126729 + gg * 0.7151522 + bb * 0.0721750;
    let z = rr * 0.0193339 + gg * 0.1191920 + bb * 0.9503041;

    return vec3<f32>(x, y, z);
}

// XYZ to Lab conversion
fn xyz_to_lab(xyz: vec3<f32>) -> vec3<f32> {
    // D65 white point
    let xn = 0.95047;
    let yn = 1.00000;
    let zn = 1.08883;

    var fx = xyz.x / xn;
    var fy = xyz.y / yn;
    var fz = xyz.z / zn;

    let delta = 6.0 / 29.0;
    let delta_sq = delta * delta;
    let delta_cube = delta_sq * delta;

    fx = select(pow(fx, 1.0 / 3.0), (fx / (3.0 * delta_sq)) + (4.0 / 29.0), fx > delta_cube);
    fy = select(pow(fy, 1.0 / 3.0), (fy / (3.0 / delta_sq)) + (4.0 / 29.0), fy > delta_cube);
    fz = select(pow(fz, 1.0 / 3.0), (fz / (3.0 * delta_sq)) + (4.0 / 29.0), fz > delta_cube);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);

    return vec3<f32>(l, a, b);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = (y * params.width + x) * params.channels;

    let r = f32(read_byte(&input, idx));
    let g = f32(read_byte(&input, idx + 1u));
    let b = f32(read_byte(&input, idx + 2u));

    let xyz = rgb_to_xyz(r, g, b);
    let lab = xyz_to_lab(xyz);

    // Store Lab values (L in [0, 100], a and b in [-128, 127] shifted to [0, 255])
    write_byte(&output, idx, u32(clamp(lab.x * 2.55, 0.0, 255.0)));  // L: 0-100 -> 0-255
    write_byte(&output, idx + 1u, u32(clamp(lab.y + 128.0, 0.0, 255.0)));  // a: -128..127 -> 0..255
    write_byte(&output, idx + 2u, u32(clamp(lab.z + 128.0, 0.0, 255.0)));  // b: -128..127 -> 0..255
}
