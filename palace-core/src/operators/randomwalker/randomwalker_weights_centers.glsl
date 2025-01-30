#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include <util.glsl>
#include <vec.glsl>
#include <size_util.glsl>
#include <tensormetadata.glsl>

layout(std430, binding = 0) readonly buffer Neighbor {
    float values[BRICK_MEM_SIZE];
} neighbor_buf[NUM_NEIGHBORS];

layout(std430, binding = 1) readonly buffer BestCenters {
    int8_t values[BRICK_MEM_SIZE][ND];
} best_centers;

layout(std430, binding = 2) readonly buffer BestCentersNeighbor {
    int8_t values[BRICK_MEM_SIZE][ND];
} best_centers_neighbor;

layout(std430, binding = 3) buffer Weights {
    float values[BRICK_MEM_SIZE][ND];
} weights;

declare_push_consts(consts);
//consts:
// dimensions
// chunk_size
// first_chunk_pos
// neighbor_chunks
// center_chunk_offset
// extent
// dim
// min_edge_weight

void init_regions(uint[ND] center_uint, out uint[ND] begin, out uint[ND] end) {
    int[ND] center = to_int(center_uint);
    int[ND] extent = fill(center, int(consts.extent));
    begin = to_uint(max(sub(center, extent), fill(center, 0)));
    end = min(to_uint(add(add(center, extent), fill(center, 1))), consts.dimensions);
}

float square(float v) {
    return v*v;
}

float sample_tensor(TensorMetaData(ND) md, uint[ND] global_pos) {
    uint[ND] chunk = chunk_pos(md, global_pos);

    uint[ND] chunk_beg = chunk_begin(md, chunk);
    uint[ND] pos_in_chunk = sub(global_pos, chunk_beg);
    uint pos_in_chunk_linear = to_linear(pos_in_chunk, md.chunk_size);

    uint[ND] neighbor_chunk_pos = sub(chunk, consts.first_chunk_pos);
    uint neighbor_linear = to_linear(neighbor_chunk_pos, consts.neighbor_chunks);

    return neighbor_buf[neighbor_linear].values[pos_in_chunk_linear];
}

void mean_and_var(uint[ND] begin, uint[ND] end, out float mean, out float var, out uint region_size_linear) {
    uint[ND] region_size = sub(end, begin);
    region_size_linear = hmul(region_size);

    TensorMetaData(ND) md;
    md.dimensions = consts.dimensions;
    md.chunk_size = consts.chunk_size;

    float sum = 0.0;
    for(int i=0; i<region_size_linear; ++i) {
        uint[ND] region_pos = from_linear(i, region_size);
        uint[ND] pos = add(region_pos, begin);

        float val = sample_tensor(md, pos);

        sum += val;
    }
    mean = sum / float(region_size_linear);

    float sum_of_diffs = 0.0;
    for(int i=0; i<region_size_linear; ++i) {
        uint[ND] region_pos = from_linear(i, region_size);
        uint[ND] pos = add(region_pos, begin);

        float val = sample_tensor(md, pos);

        sum_of_diffs += square(val - mean);
    }

    var = sum_of_diffs / float(region_size_linear-1);
}

float bhattacharyya_var_gaussian(float mean1, float mean2, float var1, float var2, uint n) {
    float nom = sqrt(var1*var2);
    float denom = (var1+var2)*0.5 + square((mean1-mean2)*0.5);

    if(denom == 0.0f) {
        return 1.0f;
    }

    float quotient = nom/denom;

    float exponent = (n-3.0)/2;
    float w = pow(quotient, exponent);

    return w;
}

int[ND] to_int(int8_t[ND] v) {
    int[ND] res;
    for(int i=0; i<ND; i+= 1) {
        res[i] = int(v[i]);
    }
    return res;
}

void main() {
    uint current_linear = global_position_linear;

    if(current_linear >= BRICK_MEM_SIZE) {
        return;
    }

    uint[ND] current = from_linear(current_linear, consts.chunk_size);

    uint dim = consts.dim;

    uint[ND] neighbor = current;
    neighbor[dim] += 1;
    if(all(less_than(add(consts.center_chunk_offset, neighbor), consts.dimensions))) {
        int8_t[ND] current_offset = best_centers.values[current_linear];

        int8_t[ND] neighbor_offset;
        if(neighbor[dim] < consts.chunk_size[dim]) {
            uint neighbor_linear = to_linear(neighbor, consts.chunk_size);
            neighbor_offset = best_centers.values[neighbor_linear];
        } else {
            uint[ND] neighbor_local = neighbor;
            neighbor_local[dim] -= consts.chunk_size[dim];
            uint neighbor_linear = to_linear(neighbor_local, consts.chunk_size);
            neighbor_offset = best_centers_neighbor.values[neighbor_linear];
        }

        int[ND] co = to_int(consts.center_chunk_offset);
        uint[ND] current_center = to_uint(add(co, add(to_int(current), to_int(current_offset))));
        uint[ND] neighbor_center = to_uint(add(co, add(to_int(neighbor), to_int(neighbor_offset))));

        uint[ND] begin1;
        uint[ND] end1;
        uint[ND] begin2;
        uint[ND] end2;

        //TODO: remove overlap
        init_regions(current_center, begin1, end1);
        init_regions(neighbor_center, begin2, end2);

        float mean1, mean2, var1, var2;
        uint n1, n2;
        mean_and_var(begin1, end1, mean1, var1, n1);
        mean_and_var(begin2, end2, mean2, var2, n2);

        float weight = bhattacharyya_var_gaussian(mean1, mean2, var1, var2, n1);

        weight = max(weight, consts.min_edge_weight);

        weights.values[current_linear][dim] = weight;
    } else {
        weights.values[current_linear][dim] = NaN;
    }
}
