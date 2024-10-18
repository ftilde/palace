#version 450

#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <vec.glsl>
#include <atomic.glsl>

#define MAX_ENTRIES_PER_ROW (2*ND + 1)

layout(std430, binding = 0) readonly buffer Input {
    float values[BRICK_MEM_SIZE];
} input_buf;

layout(std430, binding = 1) readonly buffer Seeds {
    float values[BRICK_MEM_SIZE];
} seeds_buf;

layout(std430, binding = 2) readonly buffer T2R {
    uint values[BRICK_MEM_SIZE];
} tensor_to_rows;

layout(std430, binding = 3) buffer MatValues {
    uint values[NUM_ROWS][MAX_ENTRIES_PER_ROW];
} mat_values;

layout(std430, binding = 4) buffer MatIndex {
    uint values[NUM_ROWS][MAX_ENTRIES_PER_ROW];
} mat_index;

layout(std430, binding = 5) buffer Vec {
    uint values[NUM_ROWS];
} vec;

declare_push_consts(consts)

#define MAT_INDEX_EMPTY 0xffffffff

uint get_mat_index(uint row, uint col) {
    for (uint r = 0; r < MAX_ENTRIES_PER_ROW; ++r) {
        if (mat_index.values[row][r] == col) {
            return r;
        } else if (mat_index.values[row][r] == MAT_INDEX_EMPTY) {
            //TODO: Maybe we can skip the above comparison
            if (atomicCompSwap(mat_index.values[row][r], MAT_INDEX_EMPTY, r) == MAT_INDEX_EMPTY) {
                return r;
            }
        }
    }
    return -1;
}

void mat_add(uint row, uint col, float value) {
    uint col_index = get_mat_index(row, col);
    atomic_add_float(mat_values.values[row][col_index], value);
}

void mat_assign(uint row, uint col, float value) {
    uint col_index = get_mat_index(row, col);
    mat_values.values[row][col_index] = floatBitsToUint(value);
}

float edge_weight(uint p1, uint p2) {
    float beta = 0.001;
    float diff = input_buf.values[p1] - input_buf.values[p2];
    return exp(-beta * diff * diff);
}

bool is_seed_point(uint linear_p) {
    return seeds_buf.values[linear_p] != -1;
}

void main() {
    uvec3 current_glsl = gl_GlobalInvocationID.xyz;

    uint[ND] current = from_glsl(current_glsl);

    for(int d = 0; d < ND; ++d) {
        if (current[d] >= consts.tensor_dim_in[d]) {
            return;
        }
    }

    uint current_linear = to_linear(current, consts.tensor_dim_in);

    float weight_sum = 0;

    bool current_is_seed = is_seed_point(current_linear);

    for(int dim=0; dim<ND; ++dim) {
        if(current[dim] > 0) {
            uint[ND] neighbor = current;
            neighbor[dim] -= 1;

            uint neighbor_linear = to_linear(neighbor, consts.tensor_dim_in);

            float weight = edge_weight(current_linear, neighbor_linear);

            if(is_seed_point(neighbor_linear)) {
                if(!current_is_seed) {
                    uint cur_row = tensor_to_rows.values[current_linear];

                    float to_add = weight * seeds_buf.values[neighbor_linear];
                    atomic_add_float(vec.values[cur_row], to_add);
                }
            } else {
                uint n_row = tensor_to_rows.values[neighbor_linear];
                if(!current_is_seed) {
                    uint cur_row = tensor_to_rows.values[current_linear];

                    mat_assign(cur_row, n_row, -weight);
                    mat_assign(n_row, cur_row, -weight);
                } else {
                    float to_add = weight * seeds_buf.values[current_linear];
                    atomic_add_float(vec.values[n_row], to_add);
                }

                // Update weight sum of neighbor with smaller index.
                mat_add(n_row, n_row, weight);
            }

            weight_sum += weight;
        }
    }

    if(!current_is_seed) {
        uint cur_row = tensor_to_rows.values[current_linear];
        mat_add(cur_row, cur_row, weight_sum);
    }
}
