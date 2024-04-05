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

layout(buffer_reference, std430) buffer Chunk {
    ChunkValue values[];
};

const int SAMPLE_RES_FOUND = 0;
const int SAMPLE_RES_OUTSIDE = 1;
const int SAMPLE_RES_NOT_PRESENT = 2;

/*
//#define sample_local(brick, local, vm, o) {\
//    uint local_index = to_linear((local), (vm).chunk_size);\
//    (o) = (brick).values[local_index];\
//}
*/

ChunkValue sample_local(Chunk brick, uint[3] loc, TensorMetaData(3) vm) {
    uint local_index = to_linear(loc, vm.chunk_size);
    return brick.values[local_index];
}

uint[3] offset_in_chunk(uint[3] pos, int dim, int by, uint[3] end, inout bool clamped) {
    uint[3] o = pos;
    o[dim] = min(uint(max(int(pos[dim]) + by, 0)), end[dim]-1);
    clamped = clamped || pos[dim] == o[dim];
    return o;
}

#define try_sample_with_grad(N, sample_pos_in, vm, bricks, found, sample_brick_pos_linear, value, grad) {\
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
        Chunk brick = (bricks)[(sample_brick_pos_linear)];\
        if(uint64_t(brick) == 0) {\
            (found) = SAMPLE_RES_NOT_PRESENT;\
        } else {\
            uint[N] brick_begin = mul(sample_brick, (vm).chunk_size);\
            uint[N] brick_end = min(mul(add(sample_brick, fill(sample_brick, 1)), (vm).chunk_size), (vm).dimensions);\
            uint[N] local = sub(sample_pos, brick_begin);\
            uint[N] local_end = sub(brick_end, brick_begin);\
            /*uint local_index = to_linear(local, (vm).chunk_size);\
            float v = brick.values[local_index];*/\
            ChunkValue v = sample_local(brick, local, vm);\
\
            for(int d = 0; d<N; ++d) {\
                bool clamped = false;\
                float p = sample_local(brick, offset_in_chunk(local, d,  1, local_end, clamped), vm);\
                float m = sample_local(brick, offset_in_chunk(local, d, -1, local_end, clamped), vm);\
                float div_inv = clamped ? 1.0 : 0.5;\
                grad[d] = (p-m)*div_inv;\
            }\
            (found) = SAMPLE_RES_FOUND;\
            (value) = v;\
        }\
    } else {\
        (found) = SAMPLE_RES_OUTSIDE;\
    }\
}

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
        Chunk brick = (bricks)[(sample_brick_pos_linear)];\
        if(uint64_t(brick) == 0) {\
            (found) = SAMPLE_RES_NOT_PRESENT;\
        } else {\
            uint[N] brick_begin = mul(sample_brick, (vm).chunk_size);\
            uint[N] local = sub(sample_pos, brick_begin);\
            uint local_index = to_linear(local, (vm).chunk_size);\
            ChunkValue v = brick.values[local_index];\
            (found) = SAMPLE_RES_FOUND;\
            (value) = v;\
        }\
    } else {\
        (found) = SAMPLE_RES_OUTSIDE;\
    }\
}

#endif
