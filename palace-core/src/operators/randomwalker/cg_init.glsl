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

layout(std430, binding = 2) readonly buffer X0 {
    float values[];
} x0;

layout(std430, binding = 3) readonly buffer B {
    float values[];
} b;

layout(std430, binding = 4) buffer C {
    float values[];
} c;

layout(std430, binding = 5) buffer R {
    float values[];
} r;

layout(std430, binding = 6) buffer H {
    float values[];
} h;

layout(std430, binding = 7) buffer D {
    float values[];
} d;

layout(std430, binding = 8) buffer RTH {
    float value[];
} rth;

shared float shared_sum[gl_WorkGroupSize.x];

declare_push_consts(consts);

void main() {
    uint row = global_position_linear;

    if(row < consts.num_rows) {
        float c_value;
        float clip_value = 1e-6;
        for(int i=0; i<MAX_ENTRIES_PER_ROW; ++i) {
            if(a_index.values[row][i] == row) {
                float diag_value = a_values.values[row][i];
                c_value = 1.0/max(clip_value, diag_value);
            }
        }

        float matprod;
        MAT_PROD_ROW(a_index.values, a_values.values, x0.values, row, matprod);

        c.values[row] = c_value;
        r.values[row] = b.values[row] - matprod;
        h.values[row] = c_value * r.values[row];
        d.values[row] = h.values[row];
    }

    DOT_PRODUCT_INIT(r.values, h.values, row, consts.num_rows, shared_sum, rth.value);
}
