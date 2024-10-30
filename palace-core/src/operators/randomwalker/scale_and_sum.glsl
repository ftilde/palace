#version 450

#extension GL_EXT_scalar_block_layout : require

#include <size_util.glsl>

AUTO_LOCAL_SIZE_LAYOUT;

layout(std430, binding = 0) readonly buffer X {
    float values[NUM_ROWS];
} x;

layout(std430, binding = 1) readonly buffer Y {
    float values[NUM_ROWS];
} y;

layout(std430, binding = 2) buffer Result {
    float values[NUM_ROWS];
} result;

declare_push_consts(consts);

void main() {
    uint row = global_position_linear;

    if(row >= NUM_ROWS) {
        return;
    }

    result.values[row] = consts.alpha * x.values[row] + y.values[row];
}
