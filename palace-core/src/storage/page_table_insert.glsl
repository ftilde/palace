#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#define ChunkValue float

#include <util.glsl>
//#include <sample.glsl>

layout(scalar, binding = 0) buffer OutputBuffer{
    uint64_t values[];
} root;

layout(scalar, push_constant) uniform PushConsts {
    uint64_t chunk_id;
    uint64_t buffer_addr;
} consts;

void main() {
    //uint current_linear = global_position_linear;
    uint current_linear = gl_GlobalInvocationID.x;

    if(current_linear >= 1) {
        return;
    }

    root.values[uint(consts.chunk_id)] = consts.buffer_addr;
}
