// Affine transformation shader
// Applies 2x3 affine matrix to transform image

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
    // Affine matrix [a b c; d e f]
    m00: f32,  // a
    m01: f32,  // b
    m02: f32,  // c
    m10: f32,  // d
    m11: f32,  // e
    m12: f32,  // f
    _pad0: f32,
    _pad1: f32,
}


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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.dst_width || y >= params.dst_height) {
        return;
    }

    // Transform destination coordinates to source coordinates
    let fx = f32(x);
    let fy = f32(y);

    // Inverse transformation: dst -> src
    let src_x = params.m00 * fx + params.m01 * fy + params.m02;
    let src_y = params.m10 * fx + params.m11 * fy + params.m12;

    // Check bounds
    if (src_x < 0.0 || src_x >= f32(params.src_width) - 1.0 ||
        src_y < 0.0 || src_y >= f32(params.src_height) - 1.0) {
        // Out of bounds - write black
        let dst_idx = (y * params.dst_width + x) * params.channels;
        for (var c: u32 = 0u; c < params.channels; c++) {
            write_byte(&output, dst_idx + c, 0u);
        }
        return;
    }

    // Bilinear interpolation
    let x0 = u32(floor(src_x));
    let y0 = u32(floor(src_y));
    let x1 = x0 + 1u;
    let y1 = y0 + 1u;

    let fx0 = src_x - floor(src_x);
    let fy0 = src_y - floor(src_y);
    let fx1 = 1.0 - fx0;
    let fy1 = 1.0 - fy0;

    let dst_idx = (y * params.dst_width + x) * params.channels;

    for (var c: u32 = 0u; c < params.channels; c++) {
        let v00 = f32(read_byte(&input, (y0 * params.src_width + x0) * params.channels + c));
        let v10 = f32(read_byte(&input, (y0 * params.src_width + x1) * params.channels + c));
        let v01 = f32(read_byte(&input, (y1 * params.src_width + x0) * params.channels + c));
        let v11 = f32(read_byte(&input, (y1 * params.src_width + x1) * params.channels + c));

        let value = v00 * fx1 * fy1 + v10 * fx0 * fy1 + v01 * fx1 * fy0 + v11 * fx0 * fy0;
        write_byte(&output, dst_idx + c, u32(value));
    }
}
