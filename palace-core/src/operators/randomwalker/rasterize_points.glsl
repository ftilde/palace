#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_debug_printf : enable

#include <size_util.glsl>
#include <vec.glsl>
#include <mat.glsl>
#include <randomwalker_shared.glsl>

layout(std430, binding = 0) readonly buffer FG {
    float values[NUM_POINTS_FG][ND];
} points_foreground;

layout(std430, binding = 1) readonly buffer BG {
    float values[NUM_POINTS_BG][ND];
} points_background;

layout(std430, binding = 2) buffer Seeds {
    float values[BRICK_MEM_SIZE];
} seeds;

declare_push_consts(consts);

bool at_voxel(float[ND] voxel, float[ND] to_check) {
    float[ND+1] point = to_homogeneous(to_check);
    float[ND] point_voxel = from_homogeneous(mul(consts.to_grid, point));

    point_voxel = clamp(point_voxel, fill(voxel, 0.0), sub(to_float(consts.tensor_dim_logical), fill(voxel, 1.0)));
    float[ND] diff = abs(sub(voxel, point_voxel));

    //if(voxel[2] >= 2.0) {
    //    debugPrintfEXT("voxel: %f %f %f\n", voxel[0], voxel[1], voxel[2]);
    //    debugPrintfEXT("diff: %f %f %f\n", diff[0], diff[1], diff[2]);
    //}

    bool[ND] close = less_than_equal(diff, fill(diff, 0.5));
    return all(close);
}

void main() {
    uint current_linear = global_position_linear;

    if(current_linear >= BRICK_MEM_SIZE) {
        return;
    }

    uint[ND] current_i = from_linear(current_linear, consts.tensor_dim_memory);
    float[ND] current = to_float(current_i);

    for(int i=0; i<NUM_POINTS_FG; ++i) {
        if(at_voxel(current, points_foreground.values[i])) {
            seeds.values[current_linear] = 1.0;
            return;
        }
    }

    for(int i=0; i<NUM_POINTS_BG; ++i) {
        if(at_voxel(current, points_background.values[i])) {
            seeds.values[current_linear] = 0.0;
            return;
        }
    }
    seeds.values[current_linear] = UNSEEDED;
}
