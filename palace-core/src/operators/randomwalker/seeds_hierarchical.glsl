//#extension GL_EXT_debug_printf : enable

#include <util.glsl>
#include <mat.glsl>
#include <vec.glsl>
#include <util.glsl>
#include <size_util.glsl>
#include <randomwalker_shared.glsl>

#define ChunkValue float

#define BRICK_MEM_SIZE BRICK_MEM_SIZE_IN
#include <sample.glsl>
#undef BRICK_MEM_SIZE

layout(std430, binding = 0) buffer RefBuffer {
    Chunk values[NUM_CHUNKS];
} chunks;

layout(std430, binding = 1) readonly buffer FG {
    float values[][N];
} points_foreground;

layout(std430, binding = 2) readonly buffer BG {
    float values[][N];
} points_background;

layout(std430, binding = 3) buffer OutputBuffer{
    float values[];
} outputData;

declare_push_consts(consts);

bool at_cuboid_border(uint[N] position, uint[N] size) {
    return any(equal(position, fill(position, 0))) || any(equal(add(position, fill(position, 1)), size));
}

bool at_voxel(float[N] voxel, float[N] point_voxel) {
    float[N] diff = abs(sub(voxel, point_voxel));

    bool[N] close = less_than_equal(diff, fill(diff, 0.5));
    return all(close);
}

void main() {
    uint gID = global_position_linear;

    if(gID >= hmul(consts.out_chunk_size_memory)) {
        return;
    }

    uint[N] out_chunk_pos = from_linear(gID, consts.out_chunk_size_memory);

    uint[N] global_pos = add(out_chunk_pos, consts.out_begin);
    float[N] sample_pos = mul(consts.grid_to_grid_scale, to_float(global_pos));
    //debugPrintfEXT("sample: %f %f %f\n", sample_pos[0], sample_pos[1], sample_pos[2]);
    map(N, sample_pos, sample_pos, round);

    float seed_value = UNSEEDED;

    bool inside = all(less_than_equal(global_pos, consts.out_tensor_size));

    if(inside) {
        bool at_volume_border = at_cuboid_border(global_pos, consts.out_tensor_size);
        bool at_chunk_border = at_cuboid_border(out_chunk_pos, consts.out_chunk_size_logical);

        if(at_chunk_border && !at_volume_border) {
            TensorMetaData(N) m_in;
            m_in.dimensions = consts.in_dimensions;
            m_in.chunk_size = consts.in_chunk_size;

            int res;
            uint sample_chunk_pos_linear;
            try_sample(N, sample_pos, m_in, chunks.values, res, sample_chunk_pos_linear, seed_value);

            if(res == SAMPLE_RES_FOUND) {
                // Nothing to do!
            } else if(res == SAMPLE_RES_NOT_PRESENT) {
                // This SHOULD not happen...
            } else /* SAMPLE_RES_OUTSIDE */ {
            }
        }
    }

    bool is_foreground = false;
    for(int i=0; i<consts.num_points_fg; ++i) {
        if(at_voxel(to_float(global_pos), points_foreground.values[i])) {
            is_foreground = true;
            break;
        }
    }

    bool is_background = false;
    for(int i=0; i<consts.num_points_bg; ++i) {
        if(at_voxel(to_float(global_pos), points_background.values[i])) {
            is_background = true;
            break;
        }
    }

    if (is_foreground != is_background) {
        seed_value = is_foreground ? 1.0 : 0.0;
    }
    // else: Implicity: On seed conflict, we do not write a seed value.

    outputData.values[gID] = seed_value;
}
