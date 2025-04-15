#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require

#include <util.glsl>
#include <vec.glsl>

layout(scalar, binding = 0) buffer OutputBuffer{
    vec4[2] values[];
} outputData;

layout(location = 0) in vec3 norm_pos;

declare_push_consts(consts);

void main() {
    uvec2 pos = uvec2(gl_FragCoord.xy);
    uint linear_pos = to_linear(pos, to_glsl(consts.out_mem_dim));

    vec4 color = vec4(norm_pos, 1.0);
    if(gl_FrontFacing) {
        outputData.values[linear_pos][0] = color;
    } else {
        outputData.values[linear_pos][1] = color;
    }
}
