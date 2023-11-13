struct Vec3 {
    float vals[3];
};

vec3 to_glsl_vec3(Vec3 v) {
    vec3 o;
    o.x = v.vals[2];
    o.y = v.vals[1];
    o.z = v.vals[0];
    return o;
}

struct UVec3 {
    uint vals[3];
};

uvec3 to_glsl_uvec3(UVec3 v) {
    uvec3 o;
    o.x = v.vals[2];
    o.y = v.vals[1];
    o.z = v.vals[0];
    return o;
}
