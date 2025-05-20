#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <vec.glsl>
#include <randomwalker_shared.glsl>
#include <size_util.glsl>

layout(std430, binding = 0) readonly buffer Input {
    T values[BRICK_MEM_SIZE];
} input_buf;

layout(std430, binding = 1) readonly buffer Neighbor {
    T values[BRICK_MEM_SIZE];
} neighbor_buf;

layout(std430, binding = 2) buffer WeightPairs {
    T values[BRICK_MEM_SIZE][ND][2];
} weight_pairs;

declare_push_consts(consts);

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
        T current_val = input_buf.values[current_linear];

        T neighbor_val;
        if(neighbor[dim] < consts.chunk_dim_in[dim]) {
            uint neighbor_linear = to_linear(neighbor, consts.chunk_dim_in);
            neighbor_val = input_buf.values[neighbor_linear];
        } else {
            neighbor[dim] -= consts.chunk_dim_in[dim];
            uint neighbor_linear = to_linear(neighbor, consts.chunk_dim_in);
            neighbor_val = neighbor_buf.values[neighbor_linear];
        }

        weight_pairs.values[current_linear][dim][0] = current_val;
        weight_pairs.values[current_linear][dim][1] = neighbor_val;
    } else {
        //T garbage_value = T(0);
        //weight_pairs.values[current_linear][dim][0] = garbage_value;
        //weight_pairs.values[current_linear][dim][1] = garbage_value;
    }
}
