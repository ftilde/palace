#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_nonuniform_qualifier : enable
//#extension GL_EXT_debug_printf : enable

#include <util.glsl>
#include <vec.glsl>
#include <size_util.glsl>
#include <tensormetadata.glsl>

layout(std430, binding = 0) readonly buffer Neighbor {
    float values[BRICK_MEM_SIZE][3];
} neighbor_buf[NUM_NEIGHBORS];

layout(std430, binding = 1) readonly buffer Input {
    float values[BRICK_MEM_SIZE];
} input_buf;

layout(std430, binding = 2) buffer BestCenters {
    int8_t values[BRICK_MEM_SIZE][ND];
} best_centers;

declare_push_consts(consts);

float[3] sample_neighbors(TensorMetaData(ND) md, uint[ND] global_pos) {
    uint[ND] chunk = chunk_pos(md, global_pos);

    uint[ND] chunk_beg = chunk_begin(md, chunk);
    uint[ND] pos_in_chunk = sub(global_pos, chunk_beg);
    uint pos_in_chunk_linear = to_linear(pos_in_chunk, md.chunk_size);

    uint[ND] neighbor_chunk_pos = sub(chunk, consts.first_chunk_pos);
    uint neighbor_linear = to_linear(neighbor_chunk_pos, consts.neighbor_chunks);

    return neighbor_buf[nonuniformEXT(neighbor_linear)].values[pos_in_chunk_linear];
}

float fit_quality(float value, float[3] mean_mul_add) {
    float mean = mean_mul_add[0];
    float mul = mean_mul_add[1];
    float add = mean_mul_add[2];

    float diff = value-mean;
    if(isinf(mul)) {
        // -> zero variance
        return diff == 0 ? 1.0 : NEG_INFINITY;
    } else {
        return - (diff * diff * mul + add);
    }
}

void main() {
    uint current_linear = global_position_linear;

    if(current_linear >= BRICK_MEM_SIZE) {
        return;
    }

    uint[ND] current = add(from_linear(current_linear, consts.chunk_size), consts.center_chunk_offset);
    int[ND] current_i = to_int(current);

    int[ND] best_offset;

    if(all(less_than(current, consts.dimensions))) {
        int[ND] extent = to_int(consts.extent);
        uint[ND] region_begin = to_uint(max(sub(current_i, extent), extent));
        uint[ND] region_end = to_uint(min(add(add(current_i, extent), fill(current_i, 1)), sub(to_int(consts.dimensions), extent)));

        uint[ND] region_size = sub(region_end, region_begin);
        uint region_size_linear = hmul(region_size);

        TensorMetaData(ND) md;
        md.dimensions = consts.dimensions;
        md.chunk_size = consts.chunk_size;

        float local_sample = input_buf.values[current_linear];

        uint[ND] arg_max = region_begin;
        float max_fit = NEG_INFINITY;

        for(int i=0; i<region_size_linear; ++i) {
            uint[ND] region_pos = from_linear(i, region_size);
            uint[ND] pos = add(region_pos, region_begin);

            float[3] mean_mul_add = sample_neighbors(md, pos);
            float fit_value = fit_quality(local_sample, mean_mul_add);

            if(fit_value > max_fit) {
                arg_max = pos;
                max_fit = fit_value;
            }
        }
        best_offset = sub(to_int(arg_max), current_i);

    } else {
        best_offset = fill(best_offset, 127);
    }

    for(int i=0; i<ND; ++i) {
        best_centers.values[current_linear][i] = int8_t(best_offset[i]);
    }
}
