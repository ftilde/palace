#version 450

#extension GL_EXT_scalar_block_layout : require

layout (local_size_x = LOCAL_SIZE) in;


layout(std430, binding = 0) buffer Table {
    uint values[BRICK_MEM_SIZE];
} tensor_to_vec_table;

declare_push_consts(consts);

void main() {
    uint global_id = gl_GlobalInvocationID.x;

    if(global_id >= BRICK_MEM_SIZE) {
        return;
    }


    bool is_right_block = (global_id & consts.s) != 0;
    if(is_right_block) {
        uint right_block_left_neighbor = (global_id & ~(consts.s-1)) - 1;
        tensor_to_vec_table.values[global_id] += tensor_to_vec_table.values[right_block_left_neighbor];
    }
}
