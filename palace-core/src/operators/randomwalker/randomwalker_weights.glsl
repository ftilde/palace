#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <vec.glsl>
#include <randomwalker_shared.glsl>
#include <size_util.glsl>

layout(std430, binding = 0) readonly buffer Input {
    float values[BRICK_MEM_SIZE];
} input_buf;

layout(std430, binding = 1) readonly buffer Neighbor {
    float values[BRICK_MEM_SIZE];
} neighbor_buf;

layout(std430, binding = 2) buffer Seeds {
    float values[BRICK_MEM_SIZE][ND];
} weights;

declare_push_consts(consts);

#ifdef WEIGHT_FUNCTION_BIAN_MEAN
float edge_weight(float v1, float v2) {
    float diff = v1 - v2;
    float beta = consts.diff_variance_inv * 2.0;
    return exp(-beta * diff * diff);
}
#endif

#ifdef WEIGHT_FUNCTION_GRADY
float edge_weight(float v1, float v2) {
    float diff = v1 - v2;
    return exp(-consts.grady_beta * diff * diff);
}
#endif

void main() {
    uint current_linear = global_position_linear;

    if(current_linear >= BRICK_MEM_SIZE) {
        return;
    }

    uint[ND] current = from_linear(current_linear, consts.chunk_dim_in);

    uint dim = consts.dim;

    uint[ND] neighbor = current;
    neighbor[dim] += 1;
    if(all(less_than(add(consts.chunk_begin, neighbor), consts.tensor_dim_in))) {
        float current_val = input_buf.values[current_linear];

        float neighbor_val;
        if(neighbor[dim] < consts.chunk_dim_in[dim]) {
            uint neighbor_linear = to_linear(neighbor, consts.chunk_dim_in);
            neighbor_val = input_buf.values[neighbor_linear];
        } else {
            neighbor[dim] -= consts.chunk_dim_in[dim];
            uint neighbor_linear = to_linear(neighbor, consts.chunk_dim_in);
            neighbor_val = neighbor_buf.values[neighbor_linear];
        }

        float weight = edge_weight(current_val, neighbor_val);

        weight = max(weight, consts.min_edge_weight);

        weights.values[current_linear][dim] = weight;
    } else {
        weights.values[current_linear][dim] = NaN;
    }
}
