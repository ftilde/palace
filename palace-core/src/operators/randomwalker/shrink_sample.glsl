#extension GL_EXT_scalar_block_layout : require

#include <size_util.glsl>
#include <vec.glsl>

layout(std430, binding = 0) readonly buffer Input {
    float values[BRICK_MEM_SIZE_IN];
} in_buf;

layout(std430, binding = 1) buffer Output {
    float values[BRICK_MEM_SIZE_OUT];
} out_buf;

declare_push_consts(consts);

void main() {
    uint global_id = global_position_linear;

    if(global_id >= consts.tensor_out_size) {
        return;
    }

    uint out_linear = global_id;
    uint[N] out3d = from_linear(out_linear, consts.tensor_dim_out);

    uint[N] in3d = add(out3d, consts.to_in_offset);
    uint in_linear = to_linear(in3d, consts.tensor_dim_in);

    out_buf.values[out_linear] = in_buf.values[in_linear];
}
