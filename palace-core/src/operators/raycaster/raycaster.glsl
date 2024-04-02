#version 450

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include <util.glsl>
#include <hash.glsl>
#include <sample.glsl>
#include <vec.glsl>
#include <color.glsl>


layout(buffer_reference, std430) buffer IndexType {
    BrickType values[];
};

layout(buffer_reference, std430) buffer QueryTableType {
    uint values[REQUEST_TABLE_SIZE];
};

struct LOD {
    IndexType index;
    QueryTableType queryTable;
    UVec3 dimensions;
    UVec3 chunk_size;
    Vec3 spacing;
    uint _padding;
};

layout (local_size_x = 32, local_size_y = 32) in;

layout(std430, binding = 0) buffer OutputBuffer{
    u8vec4 values[];
} output_data;

layout(scalar, binding = 1) buffer EntryExitPoints{
    vec4[2] values[];
} entry_exit_points;

layout(scalar, binding = 2) buffer LodBuffer {
    LOD levels[NUM_LEVELS];
} vol;

struct State {
    float t;
    #ifdef COMPOSITING_MIP
    float intensity;
    #elif COMPOSITING_DVR
    vec4 color;
    #endif
};

layout(std430, binding = 3) buffer StateBuffer {
    State values[];
} state_cache;

layout(std430, binding = 4) buffer TFTableBuffer {
    u8vec4 values[];
} tf_table;

declare_push_consts(consts);

struct EEPoint {
    vec3 entry;
    vec3 exit;
};

vec3 norm_to_voxel(vec3 pos, LOD l) {
    return pos * vec3(to_glsl_uvec3(l.dimensions)) - vec3(0.5);
}

vec3 voxel_to_world(vec3 pos, LOD l) {
    return pos * to_glsl_vec3(l.spacing);
}

vec3 world_to_voxel(vec3 pos, LOD l) {
    return pos / to_glsl_vec3(l.spacing);
}

vec3 norm_to_world(vec3 pos, LOD l) {
    return voxel_to_world(norm_to_voxel(pos, l), l);
}

bool sample_ee(uvec2 pos, out EEPoint eep, LOD l) {

    if(pos.x >= consts.out_mem_dim.x || pos.y >= consts.out_mem_dim.y) {
        return false;
    }

    uint gID = pos.x + pos.y * consts.out_mem_dim.x;

    vec4 entry = entry_exit_points.values[gID][0];
    vec4 exit = entry_exit_points.values[gID][1];

    eep.entry = norm_to_world(entry.xyz, l);
    eep.exit = norm_to_world(exit.xyz, l);

    return entry.a > 0.0 && exit.a > 0.0;
}

#define T_DONE (intBitsToFloat(int(0xFFC00000u)))

u8vec4 classify(float val) {
    float norm = (val-consts.tf_min)/(consts.tf_max - consts.tf_min);
    uint index = min(uint(max(0.0, norm) * consts.tf_len), consts.tf_len - 1);
    return tf_table.values[index];
}

#ifdef COMPOSITING_MIP
void update_state(inout State state, float intensity, float step_size) {
    state.intensity = max(state.intensity, intensity);
    return false;
}

u8vec4 color_from_state(State state) {
    return classify(state.intensity);
}
#endif

#ifdef COMPOSITING_DVR
void update_state(inout State state, float intensity, float step_size) {
    u8vec4 sample_u8 = classify(intensity);
    vec4 sample_f = to_uniform(sample_u8);

    // Welp, there appears to be another graphics driver bug. If the following
    // (pretty much nonsensical) lines are removed, rendering is way slower and
    // there are a few dark voxels in the volume.
    if(sample_u8.a == 255 && sample_f.a != 0.0) {
        sample_f.r *= 1.00001;
    }

    float alpha = 1.0 - pow(1.0 - sample_f.a, step_size * 200.0);

    state.color.rgb = state.color.rgb + alpha * (1.0 - state.color.a) * sample_f.rgb;
    state.color.a   = state.color.a   + alpha * (1.0 - state.color.a);

    if (state.color.a >= 0.95) {
        state.color.a = 1.0;
        state.t = T_DONE;
    }
}

u8vec4 color_from_state(State state) {
    return from_uniform(clamp(state.color, 0.0, 1.0));
    //return intensity_to_grey(state.color.a);
}
#endif

void main()
{
    uvec2 out_pos = gl_GlobalInvocationID.xy;
    uint gID = out_pos.x + out_pos.y * consts.out_mem_dim.x;

    EEPoint eep;

    if(!(out_pos.x >= consts.out_mem_dim.x || out_pos.y >= consts.out_mem_dim.y)) {

        LOD root_level = vol.levels[0];
        bool valid = sample_ee(out_pos, eep, root_level);

        u8vec4 color;
        if(valid) {
            State state = state_cache.values[gID];

            if(state.t != T_DONE) {

                EEPoint eep_x;
                if(!sample_ee(out_pos + uvec2(1, 0), eep_x, root_level)) {
                    if(!sample_ee(out_pos - uvec2(1, 0), eep_x, root_level)) {
                        eep_x.entry = vec3(0.0);
                        eep_x.exit = vec3(1.0);
                    }
                }
                EEPoint eep_y;
                if(!sample_ee(out_pos + uvec2(0, 1), eep_y, root_level)) {
                    if(!sample_ee(out_pos - uvec2(0, 1), eep_y, root_level)) {
                        eep_y.entry = vec3(0.0);
                        eep_y.exit = vec3(1.0);
                    }
                }
                vec3 neigh_x = eep_x.entry;
                vec3 neigh_y = eep_y.entry;
                vec3 center = eep.entry;
                vec3 front = eep.exit - eep.entry;

                vec3 rough_dir_x = neigh_x - center;
                vec3 rough_dir_y = neigh_y - center;
                vec3 dir_x = normalize(cross(rough_dir_y, front));
                vec3 dir_y = normalize(cross(rough_dir_x, front));

                vec3 start = eep.entry;
                vec3 end = eep.exit;
                float t_end = distance(start, end);
                vec3 dir = normalize(end - start);

                float start_pixel_dist = abs(dot(dir_x, eep_x.entry - eep.entry));
                float end_pixel_dist = abs(dot(dir_x, eep_x.exit - eep.exit));


                float lod_coarseness = consts.lod_coarseness;
                float oversampling_factor = consts.oversampling_factor;

                uint level_num = 0;
                while(state.t <= t_end) {
                    float alpha = state.t/t_end;
                    float pixel_dist = start_pixel_dist * (1.0-alpha) + end_pixel_dist * alpha;

                    while(level_num < NUM_LEVELS - 1) {
                        uint next = level_num+1;
                        vec3 next_spacing = to_glsl_vec3(vol.levels[next].spacing);
                        float left_spacing_dist = length(abs(dir_x) * next_spacing);
                        if(left_spacing_dist >= pixel_dist * lod_coarseness) {
                            break;
                        }
                        level_num = next;
                    }
                    LOD level = vol.levels[level_num];

                    TensorMetaData(3) m_in;
                    m_in.dimensions = level.dimensions.vals;
                    m_in.chunk_size = level.chunk_size.vals;

                    vec3 p = start + state.t*dir;

                    vec3 pos_voxel_g = round(world_to_voxel(p, level));
                    float[3] pos_voxel = from_glsl(pos_voxel_g);

                    float step = length(abs(dir) * to_glsl_vec3(level.spacing)) / oversampling_factor;

                    int res;
                    uint sample_brick_pos_linear;
                    float sampled_intensity;
                    try_sample(3, pos_voxel, m_in, level.index.values, res, sample_brick_pos_linear, sampled_intensity);
                    bool stop = false;
                    if(res == SAMPLE_RES_FOUND) {
                        update_state(state, sampled_intensity, step);
                    } else if(res == SAMPLE_RES_NOT_PRESENT) {
                        try_insert_into_hash_table(level.queryTable.values, REQUEST_TABLE_SIZE, sample_brick_pos_linear);
                        break;
                    } else /*res == SAMPLE_RES_OUTSIDE*/ {
                        // Should only happen at the border of the volume due to rounding errors
                    }


                    state.t += step;
                }
                if(state.t > t_end) {
                    state.t = T_DONE;
                    //if(level_num > 0) {
                    //    state.intensity = 0.0;
                    //}
                }

                state_cache.values[gID] = state;
            }

            color = color_from_state(state);
        } else {
            color = u8vec4(0);
        }

        output_data.values[gID] = color;
    }
}
