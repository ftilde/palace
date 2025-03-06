//#extension GL_EXT_debug_printf : enable

#ifndef GLSL_SAMPLE
#define GLSL_SAMPLE

#ifndef ChunkValue
#error ChunkValue is not defined
#endif

#include<page_table.glsl>
#include<vec.glsl>
#include<tensormetadata.glsl>
#include<hash.glsl>

layout(buffer_reference, std430) buffer Chunk {
    ChunkValue values[];
};

layout(buffer_reference, std430) buffer UseTableType {
    uint64_t values[];
};

const int SAMPLE_RES_FOUND = 0;
const int SAMPLE_RES_OUTSIDE = 1;
const int SAMPLE_RES_NOT_PRESENT = 2;

int try_find_chunk(PageTablePage root, uint64_t chunk_index, UseTableType use_table, uint use_table_size, out Chunk chunk) {
    //TODO NO_PUSH_main: Avoid thrashing the hash table by saving the last position and do not insert then.
    uvec3 level_indices = page_table_index_to_level_indices(chunk_index);

    PageTablePage l1 = PageTablePage(root.values[level_indices[0]]);
    if(uint64_t(l1) == 0) {
        //debugPrintfEXT("not found l1 %lu: %d\n", uint64_t(chunk), level_indices[0]);
        return SAMPLE_RES_NOT_PRESENT;
    }
    if(uint64_t(use_table) != 0) {
        try_insert_into_hash_table(use_table.values, use_table_size, uint64_t(l1));
    }

    PageTablePage l2 = PageTablePage(l1.values[level_indices[1]]);
    if(uint64_t(l2) == 0) {
        //debugPrintfEXT("not found l2 %lu: %d\n", uint64_t(chunk), level_indices[1]);
        return SAMPLE_RES_NOT_PRESENT;
    }
    if(uint64_t(use_table) != 0) {
        try_insert_into_hash_table(use_table.values, use_table_size, uint64_t(l2));
    }

    chunk = Chunk(l2.values[level_indices[2]]);
    if(uint64_t(chunk) == 0) {
        //debugPrintfEXT("not found leaf %lu: %d\n", uint64_t(chunk), level_indices[2]);
        return SAMPLE_RES_NOT_PRESENT;
    }
    if(uint64_t(use_table) != 0) {
        try_insert_into_hash_table(use_table.values, use_table_size, uint64_t(chunk));
    }

    //debugPrintfEXT("found %lu: %d %d %d \n", uint64_t(chunk), level_indices[0], level_indices[1], level_indices[2]);

    return SAMPLE_RES_FOUND;
}

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

#define try_sample_with_grad(N, sample_pos_in, vm, bricks, use_table, use_table_size, found, sample_brick_pos_linear, value, grad) {\
    int[N] sample_pos_u = to_int(sample_pos_in);\
\
    if(all(less_than_equal(fill(sample_pos_u, 0), sample_pos_u)) && all(less_than(sample_pos_u, to_int((vm).dimensions)))) {\
\
        uint[N] sample_pos = to_uint(sample_pos_u);\
        uint[N] sample_brick = div(sample_pos, (vm).chunk_size);\
        uint[N] dim_in_bricks = dim_in_bricks((vm));\
\
        (sample_brick_pos_linear) = to_linear64(sample_brick, dim_in_bricks);\
\
        Chunk brick;\
        (found) = try_find_chunk(bricks, sample_brick_pos_linear, use_table, use_table_size, brick);\
        if((found) == SAMPLE_RES_FOUND) {\
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

#define try_sample(N, sample_pos_in, vm, bricks, use_table, use_table_size, found, sample_brick_pos_linear, value) {\
    int[N] sample_pos_u = to_int(sample_pos_in);\
\
    if(all(less_than_equal(fill(sample_pos_u, 0), sample_pos_u)) && all(less_than(sample_pos_u, to_int((vm).dimensions)))) {\
\
        uint[N] sample_pos = to_uint(sample_pos_u);\
        uint[N] sample_brick = div(sample_pos, (vm).chunk_size);\
        uint[N] dim_in_bricks = dim_in_bricks((vm));\
\
        (sample_brick_pos_linear) = to_linear64(sample_brick, dim_in_bricks);\
\
        Chunk brick;\
        (found) = try_find_chunk(bricks, sample_brick_pos_linear, UseTableType(0UL), 12, brick);\
        if((found) == SAMPLE_RES_FOUND) {\
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
