#extension GL_EXT_scalar_block_layout : require

#include <randomwalker_shared.glsl>
#include <size_util.glsl>
#include <vec.glsl>

layout(std430, binding = 0) readonly buffer Seeds {
    float values[BRICK_MEM_SIZE];
} seeds_buf;

layout(std430, binding = 1) buffer Table {
    uint values[BRICK_MEM_SIZE];
} tensor_to_vec_table;

layout(std430, binding = 2) buffer NumRows {
    uint value;
} num_rows;

declare_push_consts(consts);

void main() {
    uint global_id = global_position_linear;

    if(global_id >= BRICK_MEM_SIZE) {
        return;
    }

    uint[N] pos3d = from_linear(global_id, consts.tensor_size_memory);
    bool inside = all(less_than(pos3d, consts.tensor_size_logical));

    if(global_id == BRICK_MEM_SIZE-1) {
        num_rows.value = tensor_to_vec_table.values[global_id];
    }

    if (!inside) {
        tensor_to_vec_table.values[global_id] = TENSOR_TO_VEC_TABLE_EMPTY;
    } else if (is_seed_value(seeds_buf.values[global_id])) {
        tensor_to_vec_table.values[global_id] = TENSOR_TO_VEC_TABLE_SEED;
    } else {
        // row ids start at 0 instead of 1
        tensor_to_vec_table.values[global_id] -= 1;
    }
}
