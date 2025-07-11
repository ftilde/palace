#extension GL_EXT_scalar_block_layout : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include <atomic.glsl>
#include <randomwalker_shared.glsl>
#include <size_util.glsl>
#include <vec.glsl>

layout(std430, binding = 0) readonly buffer Seeds {
    float values[BRICK_MEM_SIZE];
} seeds_buf;

layout(std430, binding = 1) buffer Table {
    uint values[BRICK_MEM_SIZE];
} tensor_to_vec_table;

const uint LOCAL_SIZE = gl_WorkGroupSize.x;
shared uint[LOCAL_SIZE] local_vals;

declare_push_consts(consts);

void main() {
    uint global_id = global_position_linear;
    uint local_id = local_index_subgroup_order;

    uint local_val;
    if(global_id < BRICK_MEM_SIZE) {
        uint[ND] pos = from_linear(global_id, consts.tensor_size_memory);

        bool inside = all(less_than(pos, consts.tensor_size_logical)) && (hmul(pos) < hmul(consts.tensor_size_logical));
        local_val = (inside && !is_seed_value(seeds_buf.values[global_id])) ? 1 : 0;
        //local_val = inside ? 1 : local_val;
    } else {
        local_val = 0;
    }

    uint subgroupAccumulated = subgroupInclusiveAdd(local_val);
    local_vals[local_id] = subgroupAccumulated;

    uint s = gl_SubgroupSize;

    while(s < LOCAL_SIZE) {
        barrier();

        bool is_right_block = (local_id & s) != 0;
        if(is_right_block) {
            uint right_block_left_neighbor = (local_id & ~(s-1)) - 1;
            local_vals[local_id] += local_vals[right_block_left_neighbor];
        }

        s *= 2;
    }

    if(global_id < BRICK_MEM_SIZE) {
        tensor_to_vec_table.values[global_id] = local_vals[local_id];
    }
}
