#ifndef VEC_GLSL
#define VEC_GLSL

#define _N 1
#include<vec_generic.glsl>
#undef _N
#define _N 2
#include<vec_generic.glsl>
#undef _N
#define _N 3
#include<vec_generic.glsl>
#undef _N
#define _N 4
#include<vec_generic.glsl>
#undef _N
#define _N 5
#include<vec_generic.glsl>
#undef _N

vec3 to_glsl(float[3] v) {
    vec3 o;
    o.x = v[2];
    o.y = v[1];
    o.z = v[0];
    return o;
}
uvec3 to_glsl(uint[3] v) {
    uvec3 o;
    o.x = v[2];
    o.y = v[1];
    o.z = v[0];
    return o;
}
float[3] from_glsl(vec3 v) {
    float[3] o;
    o[2] = v.x;
    o[1] = v.y;
    o[0] = v.z;
    return o;
}
uint[3] from_glsl(uvec3 v) {
    uint[3] o;
    o[2] = v.x;
    o[1] = v.y;
    o[0] = v.z;
    return o;
}

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

// Generic vec stuff below:
#define map(N, v, o, func)\
{\
    for(int i=0; i<N; i+= 1) {\
        o[i] = func(v[i]);\
    }\
}

#endif
