#extension GL_EXT_scalar_block_layout : require

#include <util.glsl>
#include <vec.glsl>
#include <atomic.glsl>
#include <randomwalker_shared.glsl>
#include <size_util.glsl>

layout(std430, binding = 0) readonly buffer Seeds {
    float values[BRICK_MEM_SIZE];
} seeds_buf;

layout(std430, binding = 1) readonly buffer T2R {
    uint values[BRICK_MEM_SIZE];
} tensor_to_rows;

layout(std430, binding = 2) readonly buffer Vec {
    float values[NUM_ROWS];
} results;

layout(std430, binding = 3) buffer Results {
    float values[BRICK_MEM_SIZE];
} out_buf;

void main() {
    uint current_linear = global_position_linear;

    if(current_linear >= BRICK_MEM_SIZE) {
        return;
    }

    uint result_index = tensor_to_rows.values[current_linear];

    float result_val;
    if(result_index == MAT_INDEX_EMPTY) {
        result_val = seeds_buf.values[current_linear];
    } else {
        result_val = results.values[result_index];
    }

    out_buf.values[current_linear] = result_val;
}
