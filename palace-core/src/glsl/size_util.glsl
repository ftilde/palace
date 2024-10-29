#define AUTO_LOCAL_SIZE_LAYOUT layout (local_size_x = 256, local_size_y = 1, local_size_z = 1) in

uint __global_position_linear(uvec3 workgroup_size) {
    uvec3 pos = gl_GlobalInvocationID;
    uvec3 size = gl_NumWorkGroups * workgroup_size;

    return pos.x + size.x * (pos.y + size.y *pos.z);
}

#define global_position_linear __global_position_linear(gl_WorkGroupSize)
