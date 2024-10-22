#version 450

#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <vec.glsl>
#include <atomic.glsl>
#include <randomwalker_shared.glsl>

#if N == 1
layout (local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
#elif N == 2
layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
#else
layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
#endif

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
