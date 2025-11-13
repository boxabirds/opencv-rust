// Median blur shader using sorting network for small kernels
// Supports kernel sizes 3, 5

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    kernel_size: u32,
}

// Swap two values if first > second

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

// Inline swap to avoid pointer aliasing issues in WGSL
fn swap_inline(a: u32, b: u32) -> vec2<u32> {
    return vec2<u32>(min(a, b), max(a, b));
}

// 3x3 median (9 elements) - optimal sorting network
// Rewritten to avoid pointer aliasing (not allowed in WGSL)
fn median9(values: array<u32, 9>) -> u32 {
    var v = values;
    var tmp: vec2<u32>;

    // Layer 1
    tmp = swap_inline(v[0], v[1]); v[0] = tmp.x; v[1] = tmp.y;
    tmp = swap_inline(v[3], v[4]); v[3] = tmp.x; v[4] = tmp.y;
    tmp = swap_inline(v[6], v[7]); v[6] = tmp.x; v[7] = tmp.y;

    // Layer 2
    tmp = swap_inline(v[1], v[2]); v[1] = tmp.x; v[2] = tmp.y;
    tmp = swap_inline(v[4], v[5]); v[4] = tmp.x; v[5] = tmp.y;
    tmp = swap_inline(v[7], v[8]); v[7] = tmp.x; v[8] = tmp.y;

    // Layer 3
    tmp = swap_inline(v[0], v[1]); v[0] = tmp.x; v[1] = tmp.y;
    tmp = swap_inline(v[3], v[4]); v[3] = tmp.x; v[4] = tmp.y;
    tmp = swap_inline(v[6], v[7]); v[6] = tmp.x; v[7] = tmp.y;

    // Layer 4
    tmp = swap_inline(v[0], v[3]); v[0] = tmp.x; v[3] = tmp.y;
    tmp = swap_inline(v[3], v[6]); v[3] = tmp.x; v[6] = tmp.y;

    // Layer 5
    tmp = swap_inline(v[0], v[3]); v[0] = tmp.x; v[3] = tmp.y;

    // Layer 6
    tmp = swap_inline(v[1], v[4]); v[1] = tmp.x; v[4] = tmp.y;
    tmp = swap_inline(v[4], v[7]); v[4] = tmp.x; v[7] = tmp.y;

    // Layer 7
    tmp = swap_inline(v[1], v[4]); v[1] = tmp.x; v[4] = tmp.y;

    // Layer 8
    tmp = swap_inline(v[2], v[5]); v[2] = tmp.x; v[5] = tmp.y;
    tmp = swap_inline(v[5], v[8]); v[5] = tmp.x; v[8] = tmp.y;

    // Layer 9
    tmp = swap_inline(v[2], v[5]); v[2] = tmp.x; v[5] = tmp.y;

    // Layer 10
    tmp = swap_inline(v[1], v[3]); v[1] = tmp.x; v[3] = tmp.y;
    tmp = swap_inline(v[5], v[7]); v[5] = tmp.x; v[7] = tmp.y;

    // Layer 11
    tmp = swap_inline(v[2], v[6]); v[2] = tmp.x; v[6] = tmp.y;
    tmp = swap_inline(v[4], v[6]); v[4] = tmp.x; v[6] = tmp.y;

    // Layer 12
    tmp = swap_inline(v[2], v[4]); v[2] = tmp.x; v[4] = tmp.y;

    // Layer 13
    tmp = swap_inline(v[2], v[3]); v[2] = tmp.x; v[3] = tmp.y;

    return v[4]; // Middle element
}

// 5x5 median (25 elements) - partial sorting to find median
fn median25(values: array<u32, 25>) -> u32 {
    var v = values;
    // Partial sort to get median at position 12
    for (var i: u32 = 0u; i < 13u; i++) {
        var min_idx = i;
        for (var j: u32 = i + 1u; j < 25u; j++) {
            if (v[j] < v[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            let temp = v[i];
            v[i] = v[min_idx];
            v[min_idx] = temp;
        }
    }
    return v[12];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let half_k = i32(params.kernel_size) / 2;
    let out_idx = (y * params.width + x) * params.channels;

    for (var c: u32 = 0u; c < params.channels; c++) {
        if (params.kernel_size == 3u) {
            var values: array<u32, 9>;
            var idx: u32 = 0u;

            for (var ky: i32 = -1; ky <= 1; ky++) {
                for (var kx: i32 = -1; kx <= 1; kx++) {
                    let py = clamp(i32(y) + ky, 0, i32(params.height) - 1);
                    let px = clamp(i32(x) + kx, 0, i32(params.width) - 1);
                    let src_idx = (u32(py) * params.width + u32(px)) * params.channels + c;
                    values[idx] = read_byte(&input, src_idx);
                    idx++;
                }
            }
            write_byte(&output, out_idx + c, median9(values));
        } else if (params.kernel_size == 5u) {
            var values: array<u32, 25>;
            var idx: u32 = 0u;

            for (var ky: i32 = -2; ky <= 2; ky++) {
                for (var kx: i32 = -2; kx <= 2; kx++) {
                    let py = clamp(i32(y) + ky, 0, i32(params.height) - 1);
                    let px = clamp(i32(x) + kx, 0, i32(params.width) - 1);
                    let src_idx = (u32(py) * params.width + u32(px)) * params.channels + c;
                    values[idx] = read_byte(&input, src_idx);
                    idx++;
                }
            }
            write_byte(&output, out_idx + c, median25(values));
        } else {
            // Fallback: just copy
            let src_idx = (y * params.width + x) * params.channels + c;
            write_byte(&output, out_idx + c, read_byte(&input, src_idx));
        }
    }
}
