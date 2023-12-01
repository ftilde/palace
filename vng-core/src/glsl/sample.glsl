#include<vec.glsl>

layout(buffer_reference, std430) buffer BrickType {
    float values[];
};

struct TensorMetaData {
    uint[N] dimensions;
    uint[N] chunk_size;
};

uint[N] dim_in_bricks(TensorMetaData vm) {
    return div_round_up(vm.dimensions, vm.chunk_size);
}

struct VolumeMetaData {
    uvec3 dimensions;
    uvec3 chunk_size;
};

uvec3 div_round_up3(uvec3 v1, uvec3 v2) {
    return (v1 + v2 - uvec3(1)) / v2;
}

uvec3 dim_in_bricks(VolumeMetaData vm) {
    return div_round_up3(vm.dimensions, vm.chunk_size);
}

const int SAMPLE_RES_FOUND = 0;
const int SAMPLE_RES_OUTSIDE = 1;
const int SAMPLE_RES_NOT_PRESENT = 2;

#define try_sample(sample_pos_in, vm, bricks, found, sample_brick_pos_linear, value) {\
    int[N] sample_pos_u = to_int(sample_pos_in);\
\
    if(all(less_than_equal(fill(sample_pos_u, 0), sample_pos_u)) && all(less_than(sample_pos_u, to_int((vm).dimensions)))) {\
\
        uint[N] sample_pos = to_uint(sample_pos_u);\
        uint[N] sample_brick = div(sample_pos, (vm).chunk_size);\
        uint[N] dim_in_bricks = dim_in_bricks((vm));\
\
        (sample_brick_pos_linear) = to_linear(sample_brick, dim_in_bricks);\
\
        BrickType brick = (bricks)[(sample_brick_pos_linear)];\
        float v = 0.0;\
        if(uint64_t(brick) == 0) {\
            (found) = SAMPLE_RES_NOT_PRESENT;\
        } else {\
            uint[N] brick_begin = mul(sample_brick, (vm).chunk_size);\
            uint[N] local = sub(sample_pos, brick_begin);\
            uint local_index = to_linear(local, (vm).chunk_size);\
            float v = brick.values[local_index];\
            (found) = SAMPLE_RES_FOUND;\
            (value) = v;\
        }\
    } else {\
        (found) = SAMPLE_RES_OUTSIDE;\
    }\
}

//#define sample_or_request(sample_pos, chunk_dim, dim_in_bricks) {
//}
        //uint64_t sbp = uint64_t(sample_brick_pos_linear);
        //try_insert_into_hash_table(request_table.values, REQUEST_TABLE_SIZE, sample_brick_pos_linear);
