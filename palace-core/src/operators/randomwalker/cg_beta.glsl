#extension GL_EXT_scalar_block_layout : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include <size_util.glsl>
#include <randomwalker_shared.glsl>

layout(std430, binding = 0) readonly buffer Z {
    float values[];
} z;

layout(std430, binding = 1) readonly buffer D {
    float values[];
} d;

layout(std430, binding = 2) readonly buffer C {
    float values[];
} c;

layout(std430, binding = 3) readonly buffer RTH {
    float value;
} rth;

layout(std430, binding = 4) readonly buffer DTZ {
    float value;
} dtz;

layout(std430, binding = 5) buffer X {
    float values[];
} x;

layout(std430, binding = 6) buffer R {
    float values[];
} r;

layout(std430, binding = 7) buffer H {
    float values[];
} h;

layout(std430, binding = 8) buffer RTH_P1 {
    float value[];
} rth_p1;

declare_push_consts(consts);

shared float shared_sum[gl_WorkGroupSize.x];

void main() {
    uint row = global_position_linear;

    if(row < consts.num_rows) {
        float alpha = (rth.value == dtz.value) ? 1.0 : rth.value / dtz.value;

        x.values[row] += alpha * d.values[row];
        r.values[row] -= alpha * z.values[row];
        h.values[row] = c.values[row] * r.values[row];
    }

    DOT_PRODUCT_INIT(r.values, h.values, row, consts.num_rows, shared_sum, rth_p1.value);
}
