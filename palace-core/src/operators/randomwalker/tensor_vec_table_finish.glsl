#version 450

#extension GL_EXT_scalar_block_layout : require

#include <randomwalker_shared.glsl>
#include <size_util.glsl>

AUTO_LOCAL_SIZE_LAYOUT;

layout(std430, binding = 0) readonly buffer Seeds {
    float values[BRICK_MEM_SIZE];
} seeds_buf;

layout(std430, binding = 1) buffer Table {
    uint values[BRICK_MEM_SIZE];
} tensor_to_vec_table;

layout(std430, binding = 2) buffer NUM_ROWS {
    uint value;
} num_rows;

void main() {
    uint global_id = gl_GlobalInvocationID.x;
    uint row = global_position_linear;

    if(global_id >= BRICK_MEM_SIZE) {
        return;
    }

    if(global_id == BRICK_MEM_SIZE-1) {
        num_rows.value = tensor_to_vec_table.values[global_id];
    }

    if (is_seed_value(seeds_buf.values[global_id])) {
        tensor_to_vec_table.values[global_id] = TENSOR_TO_VEC_TABLE_SEED;
    } else {
        // ids into memory start at 0 instead of 1
        tensor_to_vec_table.values[global_id] -= 1;
    }
}
