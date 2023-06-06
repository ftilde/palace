layout(buffer_reference, std430) buffer BrickType {
    float values[BRICK_MEM_SIZE];
};

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

float try_sample(uvec3 sample_pos, VolumeMetaData vm, BrickType[NUM_BRICKS] bricks, out bool found, out uint sample_brick_pos_linear) {
    uvec3 sample_brick = sample_pos / vm.chunk_size;
    uvec3 dim_in_bricks = dim_in_bricks(vm);

    sample_brick_pos_linear = to_linear3(sample_brick, dim_in_bricks);

    BrickType brick = bricks[sample_brick_pos_linear];
    float v = 0.0;
    if(uint64_t(brick) == 0) {
        found = false;
    } else {
        uvec3 brick_begin = sample_brick * vm.chunk_size;
        uvec3 local = sample_pos - brick_begin;
        uint local_index = to_linear3(local, vm.chunk_size);
        float v = brick.values[local_index];
        found = true;
        return v;
    }
}

//#define sample_or_request(sample_pos, chunk_dim, dim_in_bricks) {
//}
        //uint64_t sbp = uint64_t(sample_brick_pos_linear);
        //try_insert_into_hash_table(request_table.values, REQUEST_TABLE_SIZE, sample_brick_pos_linear);
