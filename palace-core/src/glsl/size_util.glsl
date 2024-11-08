uint _global_position_linear(uvec3 workgroup_size) {
    uvec3 pos = gl_GlobalInvocationID;
    uvec3 size = gl_NumWorkGroups * workgroup_size;

    return pos.x + size.x * (pos.y + size.y *pos.z);
}

#define global_position_linear _global_position_linear(gl_WorkGroupSize)

#define local_index_subgroup_order (gl_SubgroupInvocationID + gl_SubgroupID * gl_SubgroupSize)
