#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <vec.glsl>
#include <mat.glsl>

layout(scalar, binding = 0) buffer OutputBuffer{
    vec4[2] values[];
} eep;

declare_push_consts(consts);

void main() {
    uvec2 pos = gl_GlobalInvocationID.xy;

    uvec2 out_mem_dim = to_glsl(consts.out_mem_dim);
    if(pos.x < out_mem_dim.x && pos.y < out_mem_dim.y) {
        uint linear_pos = to_linear(pos, out_mem_dim);
        if(eep.values[linear_pos][0].w < 1.0) {
            vec2 pos_norm = vec2(pos)/vec2(out_mem_dim);
            vec4 ndc_near_plane = vec4(2.0 * pos_norm - 1.0, -1.0, 1.0);

            vec4 normalized_pos = to_glsl(consts.projection_to_norm) * ndc_near_plane;
            normalized_pos /= normalized_pos.w;

            eep.values[linear_pos][0] = normalized_pos;
        }
    }
}
