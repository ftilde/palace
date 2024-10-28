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

layout(std430, binding = 0) readonly buffer Input {
    float values[BRICK_MEM_SIZE];
} input_buf;

layout(std430, binding = 1) buffer Seeds {
    float values[BRICK_MEM_SIZE][ND];
} weights;

declare_push_consts(consts)

#ifdef WEIGHT_FUNCTION_GRADY
float edge_weight(uint p1, uint p2) {
    float diff = input_buf.values[p1] - input_buf.values[p2];
    return exp(-consts.grady_beta * diff * diff);
}
#endif

void main() {
    uvec3 current_glsl = gl_GlobalInvocationID.xyz;

    uint[ND] current = from_glsl(current_glsl);

    for(int d = 0; d < ND; ++d) {
        if (current[d] >= consts.tensor_dim_in[d]) {
            return;
        }
    }

    uint current_linear = to_linear(current, consts.tensor_dim_in);

    for(int dim=0; dim<ND; ++dim) {
        uint[ND] neighbor = current;
        neighbor[dim] += 1;
        if(neighbor[dim] < consts.tensor_dim_in[dim]) {
            uint neighbor_linear = to_linear(neighbor, consts.tensor_dim_in);

            float weight = edge_weight(current_linear, neighbor_linear);

            //TODO: make configurable
            float min_edge_weight = 0.00001;
            weight = max(weight, min_edge_weight);

            weights.values[current_linear][dim] = weight;
        } else {
            weights.values[current_linear][dim] = NaN;
        }
    }
}
