#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : require

#define ChunkValue INPUT_DTYPE

#include <util.glsl>
#include <util2d.glsl>
#include <color.glsl>
#include <hash.glsl>
#include <sample.glsl>
#include <mat.glsl>
#include <vec.glsl>

layout(scalar, binding = 0) buffer OutputBuffer{
    u8vec4 values[];
} output_data;

layout(std430, binding = 1) buffer QueryTable {
    uint64_t values[REQUEST_TABLE_SIZE];
} request_table;

layout(std430, binding = 2) buffer StateBuffer {
    uint values[];
} state;

layout(std430, binding = 3) buffer ValueBuffer{
    float values[];
} brick_values;

layout(std430, binding = 4) buffer TFTableBuffer {
    u8vec4 values[];
} tf_table;

layout(std430, binding = 5) buffer CBTQueryTable {
    uint64_t values[REQUEST_TABLE_SIZE];
} cbt_request_table;

declare_push_consts(consts);

#define UNINIT 0
#define INIT_VAL 1
#define INIT_EMPTY 2

void classify(in float val, out u8vec4 result) {
    apply_tf(tf_table.values, consts.tf_len, consts.tf_min, consts.tf_max, val, result);
}

void main()
{
    uvec2 out_pos = gl_GlobalInvocationID.xy;
    uint gID = out_pos.x + out_pos.y * consts.out_mem_dim[1];
    if(out_pos.x < consts.out_mem_dim[1] && out_pos.y < consts.out_mem_dim[0]) {
        uint s = state.values[gID];

        u8vec4 val;
        if(s == INIT_VAL) {
            float v = brick_values.values[gID];
            classify(v, val);
        } else if(s == INIT_EMPTY) {
            val = checkered_color(out_pos);
        } else {
            vec3 pos = vec3(vec2(out_pos + to_glsl(consts.out_begin)), 0);
            //vec3 sample_pos_f = mulh_mat4(transform.value, pos);
            vec3 sample_pos_f = (to_glsl(consts.transform) * vec4(pos, 1)).xyz;

            TensorMetaData(3) m_in;
            m_in.dimensions = consts.vol_dim;
            m_in.chunk_size = consts.chunk_dim;

            // Round to nearest neighbor
            // Floor+0.5 is chosen instead of round to ensure compatibility with f32::round() (see
            // test_sliceviewer below)
            vec3 sample_pos_g = floor(sample_pos_f + vec3(0.5));
            float[3] sample_pos = from_glsl(sample_pos_g);

            bool do_sample_volume = true;
            float sampled_intensity;
            int res;

            #ifdef CONST_TABLE_DTYPE
            TensorMetaData(3) const_table_m_in;
            float[3] sample_chunk_pos = div(sample_pos, to_float(m_in.chunk_size));
            const_table_m_in.dimensions = dim_in_bricks(m_in);
            const_table_m_in.chunk_size = consts.cbt_chunk_size;

            uint64_t cbt_sample_brick_pos_linear;

            CONST_TABLE_DTYPE sampled_chunk_value;
            //TODO: need to use usetable
            try_sample(3, sample_chunk_pos, const_table_m_in, PageTablePage(consts.cbt_page_table_root), UseTableType(0), 0, res, cbt_sample_brick_pos_linear, sampled_chunk_value);

            sampled_intensity = float(sampled_chunk_value);

            if(res == SAMPLE_RES_FOUND) {
                if (floatBitsToUint(sampled_chunk_value) != MARKER_NOT_CONST_BITS) {
                    //do_sample_volume = false;
                }
            } else if(res == SAMPLE_RES_NOT_PRESENT) {
                try_insert_into_hash_table(cbt_request_table.values, REQUEST_TABLE_SIZE, cbt_sample_brick_pos_linear);
                do_sample_volume = false;
            } else /*res == SAMPLE_RES_OUTSIDE*/ {
                // Should only happen at the border of the volume due to rounding errors
            }
            #endif

            if(do_sample_volume) {
                ivec3 vol_dim = ivec3(to_glsl(consts.vol_dim));

                uint64_t sample_brick_pos_linear;
                INPUT_DTYPE sampled_intensity_raw;
                try_sample(3, sample_pos, m_in, PageTablePage(consts.page_table_root), UseTableType(consts.use_table), USE_TABLE_SIZE, res, sample_brick_pos_linear, sampled_intensity_raw);

                sampled_intensity = float(sampled_intensity_raw);

                if(res == SAMPLE_RES_NOT_PRESENT) {
                    try_insert_into_hash_table(request_table.values, REQUEST_TABLE_SIZE, sample_brick_pos_linear);
                }
            }

            if(res == SAMPLE_RES_FOUND) {
                classify(sampled_intensity, val);

                state.values[gID] = INIT_VAL;
                brick_values.values[gID] = sampled_intensity;
            } else if(res == SAMPLE_RES_NOT_PRESENT) {
                val = COLOR_NOT_LOADED;
            } else /* SAMPLE_RES_OUTSIDE */ {
                val = checkered_color(out_pos);

                state.values[gID] = INIT_EMPTY;
            }
        }

        output_data.values[gID] = val;
    }
}
