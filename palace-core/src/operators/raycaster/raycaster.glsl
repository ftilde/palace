#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#define ChunkValue float

#include <util.glsl>
#include <hash.glsl>
#include <sample.glsl>
#include <vec.glsl>
#include <color.glsl>


layout(buffer_reference, std430) buffer IndexType {
    Chunk values[];
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

layout(scalar, binding = 0) buffer OutputBuffer{
    u8vec4 values[];
} output_data;

layout(scalar, binding = 1) buffer EntryExitPoints{
    vec4[2] values[];
} entry_exit_points;

layout(scalar, binding = 2) buffer LodBuffer {
    LOD levels[NUM_LEVELS];
} vol;

layout(std430, binding = 3) buffer RayStateBuffer {
    float values[];
} state_cache;

layout(scalar, binding = 4) buffer RayColorBuffer {
    u8vec4 values[];
} state_colors;

layout(std430, binding = 5) buffer TFTableBuffer {
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
    u8vec4 result;
    apply_tf(tf_table.values, consts.tf_len, consts.tf_min, consts.tf_max, val, result);
    return result;
}

#ifdef COMPOSITING_MOP
void update_state(inout float t, inout u8vec4 state_color, u8vec4 color, float step_size) {
    if(state_color.a < color.a) {
        state_color = color;
    }
}
#endif

#ifdef COMPOSITING_DVR
void update_state(inout float t, inout u8vec4 state_color, u8vec4 color, float step_size) {
    const float REFERENCE_STEP_SIZE_INV = 256.0;

    vec4 sample_f = to_uniform(color);
    vec4 state_color_f = to_uniform(state_color);

    float alpha = 1.0 - pow(1.0 - sample_f.a, step_size * REFERENCE_STEP_SIZE_INV);

    state_color_f.rgb = state_color_f.rgb + alpha * (1.0 - state_color_f.a) * sample_f.rgb;
    state_color_f.a   = state_color_f.a   + alpha * (1.0 - state_color_f.a);

    if (state_color_f.a >= 0.95) {
        state_color_f.a = 1.0;
        t = T_DONE;
    }

    state_color = from_uniform(state_color_f);
}
#endif

u8vec3 apply_phong_shading(u8vec4 sample_u8, vec3 normal, vec3 view, vec3 light) {
    vec4 sample_f = to_uniform(sample_u8);
    // Welp, there appears to be another graphics driver bug. If the following
    // (pretty much nonsensical) lines are removed, rendering is way slower and
    // there are a few dark voxels in the volume.
    if(sample_u8.a == 255 && sample_f.a != 0.0) {
        sample_f.r *= 1.00001;
    }

    vec3 ambient_light = vec3(0.2);
    vec3 diffuse_light = vec3(0.8);
    vec3 specular_light = vec3(0.5);
    float shininess = 60.0;

    vec3 color = sample_f.xyz;
    vec3 o;
    o  = color * ambient_light;
    o += color * diffuse_light * max(0.0, dot(normal, light));
    o += color * specular_light * pow(max(0.0, dot(normal, normalize(light + view))), shininess);

    return from_uniform(o);
}

void main()
{
    uvec2 out_pos = gl_GlobalInvocationID.xy;
    uint gID = out_pos.x + out_pos.y * consts.out_mem_dim.x;

    EEPoint eep;

    if(!(out_pos.x >= consts.out_mem_dim.x || out_pos.y >= consts.out_mem_dim.y)) {

        LOD root_level = vol.levels[0];
        bool valid = sample_ee(out_pos, eep, root_level);

        if(valid) {
            float t = consts.reset_state != 0 ? 0.0 : state_cache.values[gID];
            u8vec4 state_color = t == 0.0 ? u8vec4(0) : state_colors.values[gID];

            if(t != T_DONE) {

                float diag = length(vec3(to_glsl_uvec3(root_level.dimensions)) * to_glsl_vec3(root_level.spacing));

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
                while(t <= t_end) {
                    float alpha = t/t_end;
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

                    vec3 p = start + t*dir;

                    vec3 pos_voxel_g = round(world_to_voxel(p, level));
                    float[3] pos_voxel = from_glsl(pos_voxel_g);

                    float step = length(abs(dir) * to_glsl_vec3(level.spacing)) / oversampling_factor;

                    int res;
                    uint sample_brick_pos_linear;
                    float sampled_intensity;

                    #ifdef SHADING_NONE
                    try_sample(3, pos_voxel, m_in, level.index.values, res, sample_brick_pos_linear, sampled_intensity);
                    #else
                    float[3] grad_f;
                    try_sample_with_grad(3, pos_voxel, m_in, level.index.values, res, sample_brick_pos_linear, sampled_intensity, grad_f);
                    #endif


                    bool stop = false;
                    if(res == SAMPLE_RES_FOUND) {
                        u8vec4 sample_col = classify(sampled_intensity);

                        #ifdef SHADING_PHONG
                        vec3 grad = to_glsl(grad_f);
                        if(length(grad) < 0.0001) {
                            grad = normalize(vec3(1.0, 1.0, 1.0));
                        } else {
                            grad = normalize(to_glsl_vec3(level.spacing) * grad);
                        }
                        sample_col.rgb = apply_phong_shading(sample_col, grad, dir, dir);
                        #endif

                        float norm_step = step / diag;
                        update_state(t, state_color, sample_col, norm_step);

                    } else if(res == SAMPLE_RES_NOT_PRESENT) {
                        try_insert_into_hash_table(level.queryTable.values, REQUEST_TABLE_SIZE, sample_brick_pos_linear);
                        break;
                    } else /*res == SAMPLE_RES_OUTSIDE*/ {
                        // Should only happen at the border of the volume due to rounding errors
                    }


                    t += step;
                }
                if(t > t_end) {
                    t = T_DONE;
                }

                state_cache.values[gID] = t;
                state_colors.values[gID] = state_color;

                // If we have rendered anything at all (or we are done) we can
                // overwrite the actual (non-internal) frame buffer. This way
                // we get a nice update from interactive to refinement frames.
                if(t == T_DONE || state_color != u8vec4(0)) {
                    output_data.values[gID] = state_color;
                }
            }
        }
    }
}
