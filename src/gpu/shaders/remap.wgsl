@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> map_x: array<f32>;
@group(0) @binding(2) var<storage, read> map_y: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    _pad: u32,
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

fn bilinear_sample(x: f32, y: f32, c: u32) -> f32 {
    let x0 = floor(x);
    let y0 = floor(y);
    let x1 = x0 + 1.0;
    let y1 = y0 + 1.0;

    let fx = x - x0;
    let fy = y - y0;

    let ix0 = i32(x0);
    let iy0 = i32(y0);
    let ix1 = i32(x1);
    let iy1 = i32(y1);

    // Clamp to image boundaries
    let cx0 = clamp(ix0, 0, i32(params.width) - 1);
    let cy0 = clamp(iy0, 0, i32(params.height) - 1);
    let cx1 = clamp(ix1, 0, i32(params.width) - 1);
    let cy1 = clamp(iy1, 0, i32(params.height) - 1);

    let idx00 = (u32(cy0) * params.width + u32(cx0)) * params.channels + c;
    let idx10 = (u32(cy0) * params.width + u32(cx1)) * params.channels + c;
    let idx01 = (u32(cy1) * params.width + u32(cx0)) * params.channels + c;
    let idx11 = (u32(cy1) * params.width + u32(cx1)) * params.channels + c;

    let v00 = f32(read_byte(&input, idx00));
    let v10 = f32(read_byte(&input, idx10));
    let v01 = f32(read_byte(&input, idx01));
    let v11 = f32(read_byte(&input, idx11));

    let v0 = v00 * (1.0 - fx) + v10 * fx;
    let v1 = v01 * (1.0 - fx) + v11 * fx;

    return v0 * (1.0 - fy) + v1 * fy;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let map_idx = y * params.width + x;
    let src_x = map_x[map_idx];
    let src_y = map_y[map_idx];

    // Check if mapping is valid
    if (src_x < 0.0 || src_x >= f32(params.width) - 1.0 ||
        src_y < 0.0 || src_y >= f32(params.height) - 1.0) {
        // Out of bounds - set to black
        for (var c = 0u; c < params.channels; c++) {
            let idx = (y * params.width + x) * params.channels + c;
            write_byte(&output, idx, 0u);
        }
        return;
    }

    for (var c = 0u; c < params.channels; c++) {
        let value = bilinear_sample(src_x, src_y, c);
        let idx = (y * params.width + x) * params.channels + c;
        write_byte(&output, idx, u32(clamp(value, 0.0, 255.0)));
    }
}
