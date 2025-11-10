@group(0) @binding(0) var<storage, read> input1: array<u32>;
@group(0) @binding(1) var<storage, read> input2: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    width: u32,
    height: u32,
    channels: u32,
    cmp_op: u32,  // 0=EQ, 1=GT, 2=GE, 3=LT, 4=LE, 5=NE
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= params.width || y >= params.height) {
        return;
    }

    for (var c = 0u; c < params.channels; c++) {
        let idx = (y * params.width + x) * params.channels + c;
        let v1 = input1[idx];
        let v2 = input2[idx];

        var result = false;
        if (params.cmp_op == 0u) { result = v1 == v2; }       // CMP_EQ
        else if (params.cmp_op == 1u) { result = v1 > v2; }   // CMP_GT
        else if (params.cmp_op == 2u) { result = v1 >= v2; }  // CMP_GE
        else if (params.cmp_op == 3u) { result = v1 < v2; }   // CMP_LT
        else if (params.cmp_op == 4u) { result = v1 <= v2; }  // CMP_LE
        else if (params.cmp_op == 5u) { result = v1 != v2; }  // CMP_NE

        output[idx] = select(0u, 255u, result);
    }
}
