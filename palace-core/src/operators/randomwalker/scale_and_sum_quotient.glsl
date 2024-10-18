#version 450

#extension GL_EXT_scalar_block_layout : require

layout(std430, binding = 0) readonly buffer O {
    float value;
} o;

layout(std430, binding = 1) readonly buffer U {
    float value;
} u;

layout(std430, binding = 2) readonly buffer X {
    float values[NUM_ROWS];
} x;

layout(std430, binding = 3) readonly buffer Y {
    float values[NUM_ROWS];
} y;

layout(std430, binding = 4) buffer Result {
    float values[NUM_ROWS];
} result;

declare_push_consts(consts);

void main() {
    uint row = gl_GlobalInvocationID.x;

    if(row >= NUM_ROWS) {
        return;
    }

    float scale = consts.alpha * (o.value == u.value ? 1.0 : o.value/u.value);

    result.values[row] = scale * x.values[row] + y.values[row];
}
