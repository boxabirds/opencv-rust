// Histogram equalization shader
// First pass: compute histogram (reduction)
// Second pass: compute CDF and apply equalization

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cdf: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    pass: u32,  // 0 = compute histogram, 1 = compute CDF, 2 = apply equalization
    _pad: u32,
}

// Pass 0: Compute histogram
@compute @workgroup_size(16, 16)
fn compute_histogram(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.width + x;
    let value = input[idx];
    atomicAdd(&histogram[value], 1u);
}

// Pass 1: Compute CDF (prefix sum) - single workgroup
@compute @workgroup_size(256, 1, 1)
fn compute_cdf(@builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(workgroup_id) group_id: vec3<u32>) {
    var temp: array<u32, 256>;
    let tid = local_id.x;

    // Load from histogram
    temp[tid] = atomicLoad(&histogram[tid]);
    workgroupBarrier();

    // Parallel prefix sum (Blelloch scan)
    var offset = 1u;
    for (var d: u32 = 128u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            temp[bi] += temp[ai];
        }
        offset *= 2u;
    }

    if (tid == 0u) {
        temp[255] = 0u;
    }
    workgroupBarrier();

    for (var d: u32 = 1u; d < 256u; d *= 2u) {
        offset >>= 1u;
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            let t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    workgroupBarrier();

    cdf[tid] = temp[tid];
}

// Pass 2: Apply equalization
@compute @workgroup_size(16, 16)
fn apply_equalization(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    let idx = y * params.width + x;
    let value = input[idx];
    let cdf_min = cdf[0];
    let total_pixels = params.width * params.height;

    // Equalization formula: ((cdf[value] - cdf_min) / (total - cdf_min)) * 255
    let equalized = u32(f32(cdf[value] - cdf_min) / f32(total_pixels - cdf_min) * 255.0);
    output[idx] = equalized;
}
