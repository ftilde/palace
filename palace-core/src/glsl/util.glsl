#ifndef UTIL_GLSL
#define UTIL_GLSL

uint to_linear(uvec2 vec_pos, uvec2 size) {
    return vec_pos.x + size.x*vec_pos.y;
}

uvec2 from_linear(uint linear_pos, uvec2 size) {
    uvec2 vec_pos;
    vec_pos.x = linear_pos % size.x;
    linear_pos /= size.x;
    vec_pos.y = linear_pos % size.y;

    return vec_pos;
}

uint to_linear(uvec3 vec_pos, uvec3 size) {
    return vec_pos.x + size.x*(vec_pos.y + size.y*vec_pos.z);
}

uvec3 from_linear(uint linear_pos, uvec3 size) {
    uvec3 vec_pos;
    vec_pos.x = linear_pos % size.x;
    linear_pos /= size.x;
    vec_pos.y = linear_pos % size.y;
    linear_pos /= size.y;
    vec_pos.z = linear_pos;

    return vec_pos;
}

void swap(inout float v1, inout float v2) {
    float tmp = v1;
    v1 = v2;
    v2 = tmp;
}


#define NaN (intBitsToFloat(int(0xFFC00000u)))
#define POS_INFINITY (uintBitsToFloat(0x7F800000))
#define NEG_INFINITY (uintBitsToFloat(0xFF800000))

#endif
