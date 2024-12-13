#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <vec.glsl>
#include <randomwalker_shared.glsl>
#include <size_util.glsl>

#define MAX_ENTRIES_PER_ROW (2*ND + 1)

layout(std430, binding = 0) readonly buffer Weights {
    float values[BRICK_MEM_SIZE][ND];
} weights;

layout(std430, binding = 1) readonly buffer Seeds {
    float values[BRICK_MEM_SIZE];
} seeds_buf;

layout(std430, binding = 2) readonly buffer T2R {
    uint values[BRICK_MEM_SIZE];
} tensor_to_rows;

layout(std430, binding = 3) buffer MatValues {
    float values[][MAX_ENTRIES_PER_ROW];
} mat_values;

layout(std430, binding = 4) buffer MatIndex {
    uint values[][MAX_ENTRIES_PER_ROW];
} mat_index;

layout(std430, binding = 5) buffer Vec {
    float values[];
} vec;

layout(std430, binding = 6) buffer ResultVec {
    float values[];
} result_vec;


#if WITH_INIT_VALUES
layout(std430, binding = 7) readonly buffer InitValues {
    float values[];
} init_values;
#endif

declare_push_consts(consts);

uint get_mat_index(uint row, uint col) {
    for (uint r = 0; r < MAX_ENTRIES_PER_ROW; ++r) {
        uint old = mat_index.values[row][r];
        if(old == col) {
            return r;
        }
        if(old == MAT_INDEX_EMPTY) {
            mat_index.values[row][r] = col;
            return r;
        }
    }
    return -1;
}

void mat_assign(uint row, uint col, float value) {
    uint col_index = get_mat_index(row, col);
    mat_values.values[row][col_index] = value;
}

bool is_seed_point(uint linear_p) {
    return is_seed_value(seeds_buf.values[linear_p]);
}

void main() {
    uint current_linear_logical = global_position_linear;

    if(current_linear_logical >= hmul(consts.tensor_size_logical)) {
        return;
    }

    uint[ND] current = from_linear(current_linear_logical, consts.tensor_size_logical);

    float weight_sum = 0.0;
    float vec_sum = 0.0;

    uint current_linear_memory = to_linear(current, consts.tensor_size_memory);

    if(is_seed_point(current_linear_memory)) {
        return;
    }
    uint cur_row = tensor_to_rows.values[current_linear_memory];

#if WITH_INIT_VALUES
    result_vec.values[cur_row] = init_values.values[current_linear_memory];
#else
    result_vec.values[cur_row] = 0.5;
#endif

    for(int dim=ND-1; dim>=0; --dim) {
        for(int offset = -1; offset<2; offset += 2) {
        //int offset = -1;{
            int[ND] neighbor = to_int(current);
            neighbor[dim] += offset;

            int low = min(int(current[dim]), neighbor[dim]);
            int high = max(int(current[dim]), neighbor[dim]);

            if(low >= 0 && uint(high) < consts.tensor_size_logical[dim]) {
                int[ND] weight_pos = to_int(current);
                weight_pos[dim] = min(int(current[dim]), neighbor[dim]);

                uint neighbor_linear = to_linear(to_uint(neighbor), consts.tensor_size_memory);
                uint weight_pos_linear = to_linear(to_uint(weight_pos), consts.tensor_size_memory);

                float weight = weights.values[weight_pos_linear][dim];

                if(is_seed_point(neighbor_linear)) {
                    float to_add = weight * seeds_buf.values[neighbor_linear];
                    vec_sum += to_add;
                } else {
                    uint n_row = tensor_to_rows.values[neighbor_linear];

                    mat_assign(cur_row, n_row, -weight);
                }

                weight_sum += weight;
            }
        }
    }

    vec.values[cur_row] = vec_sum;
    mat_assign(cur_row, cur_row, weight_sum);
}
