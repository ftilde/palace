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

    if(global_id >= consts.overlap_size) {
        return;
    }

    uint[N] overlap3d = from_linear(global_id, consts.overlap_dim);

    uint[N] in3d = add(overlap3d, consts.to_in_offset);
    uint in_linear = to_linear(in3d, consts.tensor_dim_in);

    uint[N] out3d = add(overlap3d, consts.to_out_offset);
    uint out_linear = to_linear(out3d, consts.tensor_dim_out);

    out_buf.values[out_linear] = in_buf.values[in_linear];
}
