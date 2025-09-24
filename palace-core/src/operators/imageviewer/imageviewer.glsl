#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <util2d.glsl>
#include <color.glsl>
#include <hash.glsl>
#include <sample.glsl>
#include <mat.glsl>

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
    u8vec4 values[];
} out_values;

declare_push_consts(consts);

#define UNINIT 0
#define INIT_VAL 1
#define INIT_EMPTY 2

void main()
{
    uvec2 out_pos = gl_GlobalInvocationID.xy;
    uint gID = out_pos.x + out_pos.y * consts.out_mem_dim[1];
    if(out_pos.x < consts.out_mem_dim[1] && out_pos.y < consts.out_mem_dim[0]) {
        uint s = state.values[gID];

        u8vec4 val;
        if(s == INIT_VAL) {
            val = out_values.values[gID];
        } else if(s == INIT_EMPTY) {
            val = checkered_color(out_pos);
        } else {
            vec2 pos = vec2(out_pos + to_glsl(consts.out_begin));
            vec2 sample_pos_f = (to_glsl(consts.transform) * vec3(pos, 1)).xy;

            TensorMetaData(2) m_in;
            m_in.dimensions = consts.input_dim;
            m_in.chunk_size = consts.chunk_dim;

            vec2 sample_pos_g = floor(sample_pos_f + vec2(0.5));
            float[2] sample_pos = from_glsl(sample_pos_g);

            ivec2 input_dim = ivec2(to_glsl(consts.input_dim));


            ChunkSampleState sample_state = init_chunk_sample_state();
            try_sample(u8vec4, 2, sample_pos, m_in, PageTablePage(consts.page_table_root), UseTableType(consts.use_table), USE_TABLE_SIZE, sample_state, val);

            if(sample_state.result == SAMPLE_RES_FOUND) {
                state.values[gID] = INIT_VAL;
                out_values.values[gID] = val;
            } else if(sample_state.result == SAMPLE_RES_NOT_PRESENT) {
                try_insert_into_hash_table(request_table.values, REQUEST_TABLE_SIZE, sample_state.chunk_pos_linear);
                val = COLOR_NOT_LOADED;
            } else /* SAMPLE_RES_OUTSIDE */ {
                val = checkered_color(out_pos);

                state.values[gID] = INIT_EMPTY;
            }
        }

        output_data.values[gID] = val;
    }
}
