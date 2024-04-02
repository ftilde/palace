#ifndef MAT_GLSL
#define MAT_GLSL

#define Mat(N) float[N][N]

#define _N 1
#include <mat_generic.glsl>
#undef _N
#define _N 2
#include <mat_generic.glsl>
#undef _N
#define _N 3
#include <mat_generic.glsl>
#undef _N
#define _N 4
#include <mat_generic.glsl>
#undef _N
#define _N 5
#include <mat_generic.glsl>
#undef _N

struct Mat4 {
    mat4 inner;
};

mat4 to_mat4(Mat4 v) {
    mat4 o;
    o[0] = v.inner[3].wzyx;
    o[1] = v.inner[2].wzyx;
    o[2] = v.inner[1].wzyx;
    o[3] = v.inner[0].wzyx;
    return o;
}

vec4 mul_mat4(Mat4 m, vec4 v) {
    return (m.inner * v.wzyx).wzyx;
}

vec3 mulh_mat4(Mat4 m, vec3 v) {
    return (m.inner * vec4(1, v.zyx)).wzy;
}

#endif
