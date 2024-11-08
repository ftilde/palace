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
    float values[NUM_ROWS][MAX_ENTRIES_PER_ROW];
} mat_values;

layout(std430, binding = 4) buffer MatIndex {
    uint values[NUM_ROWS][MAX_ENTRIES_PER_ROW];
} mat_index;

layout(std430, binding = 5) buffer Vec {
    float values[NUM_ROWS];
} vec;

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
    uint current_linear = global_position_linear;

    if(current_linear >= BRICK_MEM_SIZE) {
        return;
    }

    uint[ND] current = from_linear(current_linear, consts.tensor_dim_in);

    float weight_sum = 0.0;
    float vec_sum = 0.0;

    if(is_seed_point(current_linear)) {
        return;
    }
    uint cur_row = tensor_to_rows.values[current_linear];

    for(int dim=0; dim<ND; ++dim) {
        for(int offset = -1; offset<2; offset += 2) {
        //int offset = -1;{
            int[ND] neighbor = to_int(current);
            neighbor[dim] += offset;

            int low = min(int(current[dim]), neighbor[dim]);
            int high = max(int(current[dim]), neighbor[dim]);

            if(low >= 0 && uint(high) < consts.tensor_dim_in[dim]) {
                int[ND] weight_pos = to_int(current);
                weight_pos[dim] = min(int(current[dim]), neighbor[dim]);

                uint neighbor_linear = to_linear(to_uint(neighbor), consts.tensor_dim_in);
                uint weight_pos_linear = to_linear(to_uint(weight_pos), consts.tensor_dim_in);

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
