uint to_linear2(uvec2 vec_pos, uvec2 size) {
    return vec_pos.x + size.x*vec_pos.y;
}

uvec2 from_linear2(uint linear_pos, uvec2 size) {
    uvec2 vec_pos;
    vec_pos.x = linear_pos % size.x;
    linear_pos /= size.x;
    vec_pos.y = linear_pos % size.y;

    return vec_pos;
}

uint to_linear3(uvec3 vec_pos, uvec3 size) {
    return vec_pos.x + size.x*(vec_pos.y + size.y*vec_pos.z);
}

uvec3 from_linear3(uint linear_pos, uvec3 size) {
    uvec3 vec_pos;
    vec_pos.x = linear_pos % size.x;
    linear_pos /= size.x;
    vec_pos.y = linear_pos % size.y;
    linear_pos /= size.y;
    vec_pos.z = linear_pos;

    return vec_pos;
}

#define NaN (intBitsToFloat(int(0xFFC00000u)));
