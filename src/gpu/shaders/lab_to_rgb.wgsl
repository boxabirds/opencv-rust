// Lab to RGB color space conversion

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
}

// Lab to XYZ conversion

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

fn lab_to_xyz(lab: vec3<f32>) -> vec3<f32> {
    let l = lab.x;
    let a = lab.y;
    let b = lab.z;

    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b / 200.0;

    let delta = 6.0 / 29.0;
    let delta_cube = delta * delta * delta;

    var x = select(3.0 * delta * delta * (fx - 4.0 / 29.0), pow(fx, 3.0), pow(fx, 3.0) > delta_cube);
    var y = select(3.0 * delta * delta * (fy - 4.0 / 29.0), pow(fy, 3.0), pow(fy, 3.0) > delta_cube);
    var z = select(3.0 * delta * delta * (fz - 4.0 / 29.0), pow(fz, 3.0), pow(fz, 3.0) > delta_cube);

    // D65 white point
    x *= 0.95047;
    y *= 1.00000;
    z *= 1.08883;

    return vec3<f32>(x, y, z);
}

// XYZ to RGB conversion
fn xyz_to_rgb(xyz: vec3<f32>) -> vec3<f32> {
    // Convert from XYZ to linear RGB
    var r = xyz.x *  3.2404542 + xyz.y * -1.5371385 + xyz.z * -0.4985314;
    var g = xyz.x * -0.9692660 + xyz.y *  1.8760108 + xyz.z *  0.0415560;
    var b = xyz.x *  0.0556434 + xyz.y * -0.2040259 + xyz.z *  1.0572252;

    // Apply gamma correction
    r = select(r * 12.92, 1.055 * pow(r, 1.0 / 2.4) - 0.055, r > 0.0031308);
    g = select(g * 12.92, 1.055 * pow(g, 1.0 / 2.4) - 0.055, g > 0.0031308);
    b = select(b * 12.92, 1.055 * pow(b, 1.0 / 2.4) - 0.055, b > 0.0031308);

    return vec3<f32>(r * 255.0, g * 255.0, b * 255.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = (y * params.width + x) * params.channels;

    // Unpack Lab values (L: 0-255 -> 0-100, a/b: 0-255 -> -128..127)
    let l = f32(read_byte(&input, idx)) / 2.55;
    let a = f32(read_byte(&input, idx + 1u)) - 128.0;
    let b = f32(read_byte(&input, idx + 2u)) - 128.0;

    let lab = vec3<f32>(l, a, b);
    let xyz = lab_to_xyz(lab);
    let rgb = xyz_to_rgb(xyz);

    write_byte(&output, idx, u32(clamp(rgb.x, 0.0, 255.0)));
    write_byte(&output, idx + 1u, u32(clamp(rgb.y, 0.0, 255.0)));
    write_byte(&output, idx + 2u, u32(clamp(rgb.z, 0.0, 255.0)));
}
