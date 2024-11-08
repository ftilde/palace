#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_EXT_scalar_block_layout : require

#define ChunkValue u8vec4

#include <util.glsl>
#include <color.glsl>
#include <hash.glsl>
#include <sample.glsl>

layout(scalar, binding = 0) buffer OutputBuffer{
    u8vec4 values[];
} output_data;

layout(std430, binding = 1) buffer RefBuffer {
    Chunk values[NUM_BRICKS];
} bricks;

layout(std430, binding = 2) buffer QueryTable {
    uint values[REQUEST_TABLE_SIZE];
} request_table;

layout(std430, binding = 3) buffer StateBuffer {
    uint values[];
} state;

layout(std430, binding = 4) buffer ValueBuffer{
    u8vec4 values[];
} out_values;

declare_push_consts(consts);

#define UNINIT 0
#define INIT_VAL 1
#define INIT_EMPTY 2

void main()
{
    uvec2 out_pos = gl_GlobalInvocationID.xy;
    uint gID = out_pos.x + out_pos.y * consts.out_mem_dim.x;
    if(out_pos.x < consts.out_mem_dim.x && out_pos.y < consts.out_mem_dim.y) {
        uint s = state.values[gID];

        u8vec4 val;
        if(s == INIT_VAL) {
            val = out_values.values[gID];
        } else if(s == INIT_EMPTY) {
            val = u8vec4(0, 0, 255, 255);
        } else {
            vec2 pos = vec2(out_pos + consts.out_begin);
            vec2 sample_pos_f = (consts.transform * vec3(pos, 1)).xy;

            TensorMetaData(2) m_in;
            m_in.dimensions = from_glsl(consts.input_dim);
            m_in.chunk_size = from_glsl(consts.chunk_dim);

            vec2 sample_pos_g = floor(sample_pos_f + vec2(0.5));
            float[2] sample_pos = from_glsl(sample_pos_g);

            ivec2 input_dim = ivec2(consts.input_dim);

            int res;
            uint sample_brick_pos_linear;
            try_sample(2, sample_pos, m_in, bricks.values, res, sample_brick_pos_linear, val);

            if(res == SAMPLE_RES_FOUND) {
                state.values[gID] = INIT_VAL;
                out_values.values[gID] = val;
            } else if(res == SAMPLE_RES_NOT_PRESENT) {
                try_insert_into_hash_table(request_table.values, REQUEST_TABLE_SIZE, sample_brick_pos_linear);
                val = u8vec4(255, 0, 0, 255);
            } else /* SAMPLE_RES_OUTSIDE */ {
                val = u8vec4(0, 0, 255, 255);

                state.values[gID] = INIT_EMPTY;
            }
        }

        output_data.values[gID] = val;
    }
}
