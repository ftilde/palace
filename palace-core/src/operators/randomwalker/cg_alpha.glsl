#extension GL_EXT_scalar_block_layout : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include <atomic.glsl>
#include <size_util.glsl>
#include <randomwalker_shared.glsl>

layout(std430, binding = 0) readonly buffer MatValues {
    float values[NUM_ROWS][MAX_ENTRIES_PER_ROW];
} a_values;

layout(std430, binding = 1) readonly buffer MatIndex {
    uint values[NUM_ROWS][MAX_ENTRIES_PER_ROW];
} a_index;

layout(std430, binding = 2) readonly buffer D {
    float values[NUM_ROWS];
} d;

layout(std430, binding = 3) buffer Z {
    float values[NUM_ROWS];
} z;

layout(std430, binding = 4) buffer DTZ {
    uint value;
} dtz;

//declare_push_consts(consts);

shared float shared_sum[gl_WorkGroupSize.x];

void main() {
    uint row = global_position_linear;

    if(row < NUM_ROWS) {
        float matprod;
        MAT_PROD_ROW(a_index.values, a_values.values, d.values, row, matprod);

        z.values[row] = matprod;
    }

    // Note: DOT_PRODUCT includes a barrier() so z.values is visibile to the workgroup when read.
    DOT_PRODUCT(d.values, z.values, row, NUM_ROWS, shared_sum, dtz.value);
}
