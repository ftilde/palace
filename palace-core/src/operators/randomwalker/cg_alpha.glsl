#extension GL_EXT_scalar_block_layout : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include <size_util.glsl>
#include <randomwalker_shared.glsl>

layout(std430, binding = 0) readonly buffer MatValues {
    float values[][MAX_ENTRIES_PER_ROW];
} a_values;

layout(std430, binding = 1) readonly buffer MatIndex {
    uint values[][MAX_ENTRIES_PER_ROW];
} a_index;

layout(std430, binding = 2) readonly buffer D {
    float values[];
} d;

layout(std430, binding = 3) buffer Z {
    float values[];
} z;

layout(std430, binding = 4) buffer DTZ {
    float value[];
} dtz;

declare_push_consts(consts);

shared float shared_sum[gl_WorkGroupSize.x];

void main() {
    uint row = global_position_linear;

    if(row < consts.num_rows) {
        float matprod;
        MAT_PROD_ROW(a_index.values, a_values.values, d.values, row, matprod);

        z.values[row] = matprod;
    }

    DOT_PRODUCT_INIT(d.values, z.values, row, consts.num_rows, shared_sum, dtz.value);
}
