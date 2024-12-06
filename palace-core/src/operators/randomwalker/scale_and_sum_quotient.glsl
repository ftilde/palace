#extension GL_EXT_scalar_block_layout : require

#include <size_util.glsl>

layout(std430, binding = 0) readonly buffer O {
    float value;
} o;

layout(std430, binding = 1) readonly buffer U {
    float value;
} u;

layout(std430, binding = 2) readonly buffer X {
    float values[];
} x;

layout(std430, binding = 3) readonly buffer Y {
    float values[];
} y;

layout(std430, binding = 4) buffer Result {
    float values[];
} result;

declare_push_consts(consts);

void main() {
    uint row = global_position_linear;

    if(row >= consts.num_rows) {
        return;
    }

    float scale = o.value == u.value ? 1.0 : o.value/u.value;

    result.values[row] = scale * x.values[row] + y.values[row];
}
