#ifndef GLSL_SAMPLE
#define GLSL_SAMPLE

#include<vec.glsl>

#define TensorMetaDataI(N) TensorMetaDataImpl ## N
#define TensorMetaData(N) TensorMetaDataI(N)

#define _N 1
#include <tensormetadata_generic.glsl>
#undef _N
#define _N 2
#include <tensormetadata_generic.glsl>
#undef _N
#define _N 3
#include <tensormetadata_generic.glsl>
#undef _N
#define _N 4
#include <tensormetadata_generic.glsl>
#undef _N
#define _N 5
#include <tensormetadata_generic.glsl>
#undef _N

layout(buffer_reference, std430) buffer BrickType {
    float values[];
};

const int SAMPLE_RES_FOUND = 0;
const int SAMPLE_RES_OUTSIDE = 1;
const int SAMPLE_RES_NOT_PRESENT = 2;

#define try_sample(N, sample_pos_in, vm, bricks, found, sample_brick_pos_linear, value) {\
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

#endif
