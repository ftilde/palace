#extension GL_EXT_scalar_block_layout : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#include <atomic.glsl>
#include <size_util.glsl>
#include <randomwalker_shared.glsl>

layout(std430, binding = 0) buffer X {
    float values[NUM_VALUES];
} x;

declare_push_consts(consts);

shared float shared_sum[gl_WorkGroupSize.x];

void main() {
    uint pos = global_position_linear * consts.stride;

    float val;
    if(pos < NUM_VALUES) {
        val = x.values[pos];
    } else {
        val = 0.0;
    }

    float sg_agg = subgroupAdd(val);

    if(gl_SubgroupInvocationID == 0) {
        shared_sum[gl_SubgroupID] = sg_agg;
    }

    uint s = 1;

    while(s < gl_NumSubgroups) {
        barrier();
        if(gl_SubgroupInvocationID == 0) {
            if(((gl_SubgroupID & ((2*s)-1)) == 0) && (gl_SubgroupID + s < gl_NumSubgroups)) {
                shared_sum[gl_SubgroupID] += shared_sum[gl_SubgroupID+s];
            }
        }
        s *= 2;
    }

    barrier();

    if(local_index_subgroup_order == 0) {
        x.values[pos] = shared_sum[0];
    }
}
