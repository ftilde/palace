#version 450

#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <vec.glsl>
#include <atomic.glsl>

layout(std430, binding = 0) readonly buffer Seeds {
    float values[BRICK_MEM_SIZE];
} seeds_buf;

layout(std430, binding = 1) readonly buffer T2R {
    uint values[BRICK_MEM_SIZE];
} tensor_to_rows;

layout(std430, binding = 2) readonly buffer Vec {
    float values[NUM_ROWS];
} results;

layout(std430, binding = 3) buffer Results {
    float values[BRICK_MEM_SIZE];
} out_buf;

declare_push_consts(consts)

#define MAT_INDEX_EMPTY 0xffffffff

void main() {
    uvec3 current_glsl = gl_GlobalInvocationID.xyz;

    uint[ND] current = from_glsl(current_glsl);

    for(int d = 0; d < ND; ++d) {
        if (current[d] >= consts.tensor_dim[d]) {
            return;
        }
    }

    uint current_linear = to_linear(current, consts.tensor_dim);

    uint result_index = tensor_to_rows.values[current_linear];

    float result_val;
    if(result_index == MAT_INDEX_EMPTY) {
        result_val = seeds_buf.values[current_linear];
    } else {
        result_val = results.values[result_index];
    }

    out_buf.values[current_linear] = result_val;
}
