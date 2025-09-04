//#extension GL_EXT_debug_printf : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <mat.glsl>
#include <vec.glsl>
#include <util.glsl>
#include <size_util.glsl>
#include <randomwalker_shared.glsl>
#include <atomic.glsl>

#define ChunkValue float

#define BRICK_MEM_SIZE BRICK_MEM_SIZE_IN
#include <sample.glsl>
#undef BRICK_MEM_SIZE

layout(std430, binding = 0) buffer OutputBuffer{
    float values[];
} outputData;

layout(std430, binding = 1) buffer MinMaxBuffer{
    uint min;
    uint max;
} min_max;

layout(std430, binding = 2) readonly buffer FG {
    float values[][N];
} points_foreground;

layout(std430, binding = 3) readonly buffer BG {
    float values[][N];
} points_background;

declare_push_consts(consts);

bool at_cuboid_border(uint[N] position, uint[N] size) {
    return any(equal(position, fill(position, 0))) || any(equal(add(position, fill(position, 1)), size));
}
bool inside_cuboid(float[N] position, float[N] lower, float[N] upper) {
    return all(less_than_equal(lower, position)) && all(less_than(position, upper));
}

bool at_voxel(float[N] voxel, float[N] point_voxel) {
    float[N] diff = abs(sub(voxel, point_voxel));

    float[N] p05 = fill(diff, 0.5);
    bool[N] close = and(less_than(neg(p05), diff), less_than_equal(diff, p05));
    return all(close);
}

shared uint shared_min;
shared uint shared_max;

const uint LOCAL_SIZE = gl_WorkGroupSize.x * gl_WorkGroupSize.y * gl_WorkGroupSize.z;
shared float[LOCAL_SIZE][N] shared_positions;
shared uint inside_batch_count;

void main() {
    uint gID = global_position_linear;
    uint lID = local_index_subgroup_order;
    if(lID == 0) {
        shared_min = floatBitsToUint(1.0);
        shared_max = floatBitsToUint(0.0);
    }
    barrier();

    float min_val = 1.0;
    float max_val = 0.0;

    if(gID < hmul(consts.out_chunk_size_memory)) {

        uint[N] out_chunk_pos = from_linear(gID, consts.out_chunk_size_memory);

        uint[N] global_pos = add(out_chunk_pos, consts.out_begin);
        float[N] sample_pos = mul(consts.grid_to_grid_scale, to_float(global_pos));
        //debugPrintfEXT("sample: %f %f %f\n", sample_pos[0], sample_pos[1], sample_pos[2]);
        map(N, sample_pos, sample_pos, round);

        float seed_value = UNSEEDED;

        bool inside = all(less_than(global_pos, consts.out_tensor_size));

        if(inside) {
            bool at_volume_border = at_cuboid_border(global_pos, consts.out_tensor_size);
            bool at_chunk_border = at_cuboid_border(out_chunk_pos, consts.out_chunk_size_logical);

            if(at_chunk_border && !at_volume_border) {
                TensorMetaData(N) m_in;
                m_in.dimensions = consts.in_dimensions;
                m_in.chunk_size = consts.in_chunk_size;

                PageTablePage page_table_root = PageTablePage(consts.page_table_root);

                ChunkSampleState sample_state = init_chunk_sample_state();
                try_sample(N, sample_pos, m_in, page_table_root, UseTableType(0UL), 0, sample_state, seed_value);

                if(sample_state.result == SAMPLE_RES_FOUND) {
                    min_val = min(min_val, seed_value);
                    max_val = max(max_val, seed_value);
                } else if(sample_state.result == SAMPLE_RES_NOT_PRESENT) {
                    // This SHOULD not happen...
                } else /* SAMPLE_RES_OUTSIDE */ {
                }

            }
        }
        bool is_foreground = false;
        bool is_background = false;

        float[N] p05 = fill(sample_pos, 0.5);
        float[N] out_begin_f = sub(to_float(consts.out_begin), p05);
        float[N] out_end_f = add(to_float(add(consts.out_begin, consts.out_chunk_size_logical)), p05);
        for(uint i=0; i<consts.num_points_fg; i+=LOCAL_SIZE) {
            if(lID == 0) {
                inside_batch_count = 0;
            }

            barrier();
            uint j = i+lID;
            if(j < consts.num_points_fg) {
                float[N] seed_point_f = points_foreground.values[j];
                if(inside_cuboid(seed_point_f, out_begin_f, out_end_f)) {
                    uint shared_i = atomicAdd(inside_batch_count, 1u);
                    shared_positions[shared_i] = seed_point_f;
                }
            }
            barrier();
            for(uint k=0; k<inside_batch_count; ++k) {
                if(at_voxel(to_float(global_pos), shared_positions[k])) {
                    is_foreground = true;
                    max_val = max(max_val, 1.0);
                    break;
                }
            }
            barrier();
        }

        for(uint i=0; i<consts.num_points_bg; i+=LOCAL_SIZE) {
            if(lID == 0) {
                inside_batch_count = 0;
            }

            barrier();
            uint j = i+lID;
            if(j < consts.num_points_bg) {
                float[N] seed_point_f = points_background.values[j];
                if(inside_cuboid(seed_point_f, out_begin_f, out_end_f)) {
                    uint shared_i = atomicAdd(inside_batch_count, 1u);
                    shared_positions[shared_i] = seed_point_f;
                }
            }
            barrier();
            for(uint k=0; k<inside_batch_count; ++k) {
                if(at_voxel(to_float(global_pos), shared_positions[k])) {
                    is_background = true;
                    min_val = min(min_val, 0.0);
                    break;
                }
            }
            barrier();
        }

        if (is_foreground != is_background) {
            seed_value = is_foreground ? 1.0 : 0.0;
        }
        // else: Implicity: On seed conflict, we do not write a seed value.

        outputData.values[gID] = seed_value;
    }

    float sg_min = subgroupMin(min_val);
    float sg_max = subgroupMax(max_val);

    if(gl_SubgroupInvocationID == 0) {
        atomic_min_float(shared_min, sg_min);
        atomic_max_float(shared_max, sg_max);
    }

    barrier();

    if(lID == 0) {
        atomic_min_float(min_max.min, uintBitsToFloat(shared_min));
        atomic_max_float(min_max.max, uintBitsToFloat(shared_max));
    }
}
