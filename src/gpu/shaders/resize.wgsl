// Bilinear Resize Shader
// GPU-accelerated image resizing with bilinear interpolation

struct ResizeParams {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    channels: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: ResizeParams;

// Bilinear interpolation

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

    let v00 = f32(read_byte(&input, idx00));
    let v10 = f32(read_byte(&input, idx10));
    let v01 = f32(read_byte(&input, idx01));
    let v11 = f32(read_byte(&input, idx11));

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
        write_byte(&output, out_idx, u32(clamp(value, 0.0, 255.0)));
    }
}
