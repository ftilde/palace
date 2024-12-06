#extension GL_EXT_scalar_block_layout : require

#include <size_util.glsl>
#include <vec.glsl>
#include <mat.glsl>

layout(std430, binding = 0) readonly buffer In {
    float values[MEM_SIZE][ND];
} points_in;

layout(std430, binding = 1) buffer Out {
    float values[MEM_SIZE][ND];
} points_out;

declare_push_consts(consts);

void main() {
    uint current_linear = global_position_linear;

    if(current_linear >= MEM_SIZE) {
        return;
    }

    float[ND] point_orig = points_in.values[current_linear];
    float[ND+1] point = to_homogeneous(point_orig);
    points_out.values[current_linear] = from_homogeneous(mul(consts.matrix, point));
}
