#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <vec.glsl>
#include <randomwalker_shared.glsl>
#include <size_util.glsl>

layout(std430, binding = 0) readonly buffer Input {
    float values[BRICK_MEM_SIZE];
} input_buf;

layout(std430, binding = 1) buffer Seeds {
    float values[BRICK_MEM_SIZE][ND];
} weights;

declare_push_consts(consts);

#ifdef WEIGHT_FUNCTION_BIAN_MEAN
float edge_weight(uint p1, uint p2) {
    float diff = input_buf.values[p1] - input_buf.values[p2];
    float beta = consts.diff_variance_inv * 2.0;
    return exp(-beta * diff * diff);
}
#endif

#ifdef WEIGHT_FUNCTION_GRADY
float edge_weight(uint p1, uint p2) {
    float diff = input_buf.values[p1] - input_buf.values[p2];
    return exp(-consts.grady_beta * diff * diff);
}
#endif

void main() {
    uint current_linear = global_position_linear;

    if(current_linear >= BRICK_MEM_SIZE) {
        return;
    }

    uint[ND] current = from_linear(current_linear, consts.tensor_dim_in);

    for(int dim=0; dim<ND; ++dim) {
        uint[ND] neighbor = current;
        neighbor[dim] += 1;
        if(neighbor[dim] < consts.tensor_dim_in[dim]) {
            uint neighbor_linear = to_linear(neighbor, consts.tensor_dim_in);

            float weight = edge_weight(current_linear, neighbor_linear);

            weight = max(weight, consts.min_edge_weight);

            weights.values[current_linear][dim] = weight;
        } else {
            weights.values[current_linear][dim] = NaN;
        }
    }
}
