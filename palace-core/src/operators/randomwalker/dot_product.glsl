#version 450

#extension GL_EXT_scalar_block_layout : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include <atomic.glsl>

layout (local_size_x = 1024) in;

layout(std430, binding = 0) readonly buffer X {
    float values[NUM_ROWS];
} x;

layout(std430, binding = 1) readonly buffer Y {
    float values[NUM_ROWS];
} y;

layout(std430, binding = 2) buffer Result {
    uint value;
} result;

//declare_push_consts(consts);

shared uint shared_sum;

void main() {
    uint row = gl_GlobalInvocationID.x;

    if(gl_LocalInvocationIndex == 0) {
        shared_sum = floatBitsToUint(0.0);
    }
    barrier();

    float val;
    if(row < NUM_ROWS) {
        val = x.values[row] * y.values[row];
    } else {
        val = 0.0;
    }

    float sg_agg = subgroupAdd(val);

    if(gl_SubgroupInvocationID == 0) {
        atomic_add_float(shared_sum, sg_agg);
    }

    barrier();

    if(gl_LocalInvocationIndex == 0) {
        atomic_add_float(result.value, uintBitsToFloat(shared_sum));
    }
}
