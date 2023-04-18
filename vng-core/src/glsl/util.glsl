uint to_linear3(uvec3 vec_pos, uvec3 size) {
    return vec_pos.x + size.x*(vec_pos.y + size.y*vec_pos.z);
}
