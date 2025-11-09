// Binary Threshold Shader
// GPU-accelerated binary thresholding

struct ThresholdParams {
    width: u32,
    height: u32,
    channels: u32,
    threshold: u32,
    max_value: u32,
    _padding: array<u32, 3>, // Align to 16 bytes
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: ThresholdParams;

@compute @workgroup_size(16, 16)
fn threshold_binary(@builtin(global_invocation_id) id: vec3<u32>) {
    let x = id.x;
    let y = id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    // Process each channel
    for (var ch = 0u; ch < params.channels; ch++) {
        let idx = (x + y * params.width) * params.channels + ch;
        let value = input[idx];

        // Binary threshold: value > threshold ? max_value : 0
        let result = select(0u, params.max_value, value > params.threshold);
        output[idx] = result;
    }
}
