#extension GL_EXT_scalar_block_layout : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include <size_util.glsl>
#include <randomwalker_shared.glsl>

layout(std430, binding = 0) readonly buffer X {
    float values[NUM_ROWS];
} x;

layout(std430, binding = 1) readonly buffer Y {
    float values[NUM_ROWS];
} y;

layout(std430, binding = 2) buffer Result {
    float value[];
} result;

//declare_push_consts(consts);

shared float shared_sum[gl_WorkGroupSize.x];

void main() {
    uint row = global_position_linear;
    DOT_PRODUCT_INIT(x.values, y.values, row, NUM_ROWS, shared_sum, result.value);
}
