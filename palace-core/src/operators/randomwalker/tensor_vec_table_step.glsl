#extension GL_EXT_scalar_block_layout : require

#include <size_util.glsl>

layout(std430, binding = 0) buffer Table {
    uint values[BRICK_MEM_SIZE];
} tensor_to_vec_table;

declare_push_consts(consts);

void main() {
    uint global_id = global_position_linear;

    uint half_mask = consts.s-1;
    uint id_lower_half = global_id & half_mask;
    uint id_upper_half = global_id & (~half_mask);
    uint id = (id_upper_half << 1) | consts.s | id_lower_half ;

    if(id >= BRICK_MEM_SIZE) {
        return;
    }

    uint right_block_left_neighbor = (id & ~half_mask) - 1;
    tensor_to_vec_table.values[id] += tensor_to_vec_table.values[right_block_left_neighbor];
}
